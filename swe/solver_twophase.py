"""Two-phase (solid + fluid) depth-averaged solver on the active-cell grid.

Extends solver_sparse.py: the solid phase (rock + ice, depth hs, velocity us) and the fluid
phase (water + fines, depth hf, velocity uf) are two shallow layers that share one free
surface eta = z + hs + hf. Each phase is driven by -g h_p grad(eta) (isotropic stress,
Pitman & Le 2005 with k = 1), so the layer-p momentum equation is the two-layer form
    d(h_p u_p)/dt + div(h_p u_p u_p + 1/2 g h_p^2) = -g h_p grad(z + h_q)
which is discretised with the same depth-limited tilt + hydrostatic reconstruction as the
single-phase code, using z + h_q as the effective bed of phase p and the total depth for the
HLL wave speeds. Phase coupling: linear drag with terminal velocity U_T (Pudasaini 2012),
solved as an exact relaxation of the velocity difference; melt moves mass from solid to
fluid; entrainment feeds solid (fraction erosion_c) and fluid.
Solid friction is Coulomb on the submerged weight, (1 - gamma) mu (1 - p) g hs, where p is
the excess pore-pressure ratio carried with the solid phase and dissipating on the
consolidation timescale h^2 / D (Iverson & George 2014). Fluid friction is Manning.
Outputs keep the single-phase conventions: h = hs + hf, hc = hs, hu = hs us + hf uf.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from solver_sparse import SparseSWE, G, LAT


class TwoPhaseSWE(SparseSWE):
    def __init__(self, z2d, active, cell, *, device, mu_s, n_w, n_d, gamma=0.4, u_t=1.0, pore_d=0.0, pore_ent=1.0,
                 h_dry=1e-3, cfl=0.45):
        super().__init__(z2d, active, cell, device=device, mu_s=mu_s, n_w=n_w, n_d=n_d, h_dry=h_dry, cfl=cfl)
        self.gamma, self.u_t, self.pore_d, self.pore_ent = gamma, u_t, pore_d, pore_ent
        zero = lambda: torch.zeros(self.n, dtype=torch.float32, device=self.dev)
        # solid phase: hs, hsu, hsv, hi (ice), hp (pore-pressure ratio * hs); fluid: hf, hfu, hfv, hq (heat)
        self.hs, self.hsu, self.hsv, self.hp = zero(), zero(), zero(), zero()
        self.hf, self.hfu, self.hfv = zero(), zero(), zero()

    # --- single-phase views used by run() and the output format ---
    @property
    def h(self): return self.hs + self.hf
    @property
    def hc(self): return self.hs
    @property
    def hu(self): return self.hsu + self.hfu
    @property
    def hv(self): return self.hsv + self.hfv
    @h.setter
    def h(self, v): pass
    @hc.setter
    def hc(self, v): pass
    @hu.setter
    def hu(self, v): pass
    @hv.setter
    def hv(self, v): pass

    def phase_velocities(self):
        inv_s = torch.where(self.hs > self.h_dry, 1.0 / self.hs.clamp_min(self.h_dry), torch.zeros_like(self.hs))
        inv_f = torch.where(self.hf > self.h_dry, 1.0 / self.hf.clamp_min(self.h_dry), torch.zeros_like(self.hf))
        return self.hsu * inv_s, self.hsv * inv_s, self.hfu * inv_f, self.hfv * inv_f

    def max_dt(self) -> float:
        us, vs, uf, vf = self.phase_velocities()
        s = torch.maximum(torch.maximum(us.abs(), vs.abs()), torch.maximum(uf.abs(), vf.abs())) + torch.sqrt(2.0 * G * self.h.clamp_min(0.0))
        smax = float(s.max())
        return 5.0 if smax <= 1e-6 else min(self.cfl * self.dx / smax, 5.0)

    @staticmethod
    def _hll2(hL, hR, uL, uR, pL, pR, htL, htR):
        """HLL for one phase with its own pressure p; wave speeds from the total depth (shared free surface)."""
        cL_ = torch.sqrt(G * htL); cR_ = torch.sqrt(G * htR)
        sL = torch.minimum(uL - cL_, uR - cR_); sR = torch.maximum(uL + cL_, uR + cR_)
        fL0, fR0 = hL * uL, hR * uR
        fL1 = hL * uL * uL + pL; fR1 = hR * uR * uR + pR
        denom = sR - sL; safe = denom.abs() > 1e-8
        w = torch.where(safe, 1.0 / torch.where(safe, denom, torch.ones_like(denom)), torch.zeros_like(denom))
        f0 = (sR * fL0 - sL * fR0 + sL * sR * (hR - hL)) * w
        f1 = (sR * fL1 - sL * fR1 + sL * sR * (hR * uR - hL * uL)) * w
        f0 = torch.where(sL >= 0, fL0, torch.where(sR <= 0, fR0, f0))
        f1 = torch.where(sL >= 0, fL1, torch.where(sR <= 0, fR1, f1))
        return f0, f1, f0 >= 0

    def _axis2(self, un_s, ut_s, un_f, ut_f, alpha, ice, pore, heat, zplus, zminus, L, R, tilt_ok):
        """Face fluxes along one axis for both phases.

        The total depth is reconstructed (tilt + hydrostatic) on the bed as in the single-phase code;
        the solid phase carries the conservative granular pressure (1-gamma) alpha g h^2/2 plus the
        buoyancy share gamma*alpha of the total pressure force; the fluid carries (1-alpha) of it.
        Returns face fluxes and the per-cell total pressure force Fp (well balanced at rest).
        """
        hs, hf, z = self.hs, self.hf, self.z
        h = hs + hf; g1 = 1.0 - self.gamma
        tilt = torch.clamp(0.5 * (zplus - zminus), -h, h) * tilt_ok
        zP = z + tilt; zM = z - tilt; hE = h - tilt; hW = h + tilt
        zf = torch.maximum(zP[L], zM[R])
        hL = (hE[L] + zP[L] - zf).clamp_min(0.0); hR = (hW[R] + zM[R] - zf).clamp_min(0.0)
        corrL = 0.5 * G * (hE[L] ** 2 - hL ** 2); corrR = 0.5 * G * (hW[R] ** 2 - hR ** 2)
        src = 0.5 * G * (hE + hW) * (zM - zP)
        aL, aR = alpha[L], alpha[R]
        # solid: granular pressure inside the Riemann solver
        f0s, f1s, ups = self._hll2(aL * hL, aR * hR, un_s[L], un_s[R], g1 * aL * 0.5 * G * hL * hL, g1 * aR * 0.5 * G * hR * hR, hL, hR)
        # fluid: advective flux only; its pressure share comes from Fp
        f0f, f1f, upf = self._hll2((1.0 - aL) * hL, (1.0 - aR) * hR, un_f[L], un_f[R], torch.zeros_like(hL), torch.zeros_like(hR), hL, hR)
        pick_s = lambda a: torch.where(ups, a[L], a[R]); pick_f = lambda a: torch.where(upf, a[L], a[R])
        pface = 0.25 * G * (hL * hL + hR * hR)
        Fp = torch.zeros_like(h)
        Fp.index_add_(0, L, -(pface + corrL)); Fp.index_add_(0, R, pface + corrR); Fp += src
        return (f0s, f1s + g1 * aL * corrL, f1s + g1 * aR * corrR, f0s * pick_s(ut_s), f0s * pick_s(ice), f0s * pick_s(pore), g1 * alpha * src,
                f0f, f1f, f1f, f0f * pick_f(ut_f), f0f * pick_f(heat), torch.zeros_like(h), Fp)

    def step(self, dt: float) -> None:
        hs, hf = self.hs, self.hf
        us, vs, uf, vf = self.phase_velocities()
        inv_s = torch.where(hs > self.h_dry, 1.0 / hs.clamp_min(self.h_dry), torch.zeros_like(hs))
        inv_f = torch.where(hf > self.h_dry, 1.0 / hf.clamp_min(self.h_dry), torch.zeros_like(hf))
        ice = (self.hi * inv_s).clamp(0.0, 1.0); pore = (self.hp * inv_s).clamp(0.0, 1.0); heat = self.hq * inv_f
        h = hs + hf
        alpha = torch.where(h > self.h_dry, hs / h.clamp_min(self.h_dry), torch.zeros_like(h)).clamp(0.0, 1.0)
        z = self.z
        zx_plus = 0.5 * (z + z[self._nbr_r]); zx_minus = 0.5 * (z + z[self._nbr_l])
        zy_plus = 0.5 * (z + z[self._nbr_d]); zy_minus = 0.5 * (z + z[self._nbr_u])
        acc = [torch.zeros_like(hs) for _ in range(9)]  # dhs, dhsu, dhsv, dhi, dhp, dhf, dhfu, dhfv, dhq
        # x faces
        r = self._axis2(us, vs, uf, vf, alpha, ice, pore, heat, zx_plus, zx_minus, self.fxL, self.fxR, self.tilt_ok_x)
        L, R = self.fxL, self.fxR
        f0s, f1sl, f1sr, f2s, f3s, f4s, srcs, f0f, f1fl, f1fr, f2f, f3f, srcf, Fp = r
        acc[0].index_add_(0, L, -f0s); acc[0].index_add_(0, R, f0s)
        acc[1].index_add_(0, L, -f1sl); acc[1].index_add_(0, R, f1sr); acc[1] += srcs + self.gamma * alpha * Fp
        acc[2].index_add_(0, L, -f2s); acc[2].index_add_(0, R, f2s)
        acc[3].index_add_(0, L, -f3s); acc[3].index_add_(0, R, f3s)
        acc[4].index_add_(0, L, -f4s); acc[4].index_add_(0, R, f4s)
        acc[5].index_add_(0, L, -f0f); acc[5].index_add_(0, R, f0f)
        acc[6].index_add_(0, L, -f1fl); acc[6].index_add_(0, R, f1fr); acc[6] += (1.0 - alpha) * Fp
        acc[7].index_add_(0, L, -f2f); acc[7].index_add_(0, R, f2f)
        acc[8].index_add_(0, L, -f3f); acc[8].index_add_(0, R, f3f)
        # y faces (normal velocity -v)
        r = self._axis2(-vs, us, -vf, uf, alpha, ice, pore, heat, zy_plus, zy_minus, self.fyL, self.fyR, self.tilt_ok_y)
        L, R = self.fyL, self.fyR
        f0s, f1sl, f1sr, f2s, f3s, f4s, srcs, f0f, f1fl, f1fr, f2f, f3f, srcf, Fp = r
        acc[0].index_add_(0, L, -f0s); acc[0].index_add_(0, R, f0s)
        acc[2].index_add_(0, L, f1sl); acc[2].index_add_(0, R, -f1sr); acc[2] -= srcs + self.gamma * alpha * Fp
        acc[1].index_add_(0, L, -f2s); acc[1].index_add_(0, R, f2s)
        acc[3].index_add_(0, L, -f3s); acc[3].index_add_(0, R, f3s)
        acc[4].index_add_(0, L, -f4s); acc[4].index_add_(0, R, f4s)
        acc[5].index_add_(0, L, -f0f); acc[5].index_add_(0, R, f0f)
        acc[7].index_add_(0, L, f1fl); acc[7].index_add_(0, R, -f1fr); acc[7] -= (1.0 - alpha) * Fp
        acc[6].index_add_(0, L, -f2f); acc[6].index_add_(0, R, f2f)
        acc[8].index_add_(0, L, -f3f); acc[8].index_add_(0, R, f3f)
        # walls at missing neighbours: total hydrostatic pressure shared by phase fraction
        pw = 0.5 * G * h * h
        acc[1] += alpha * pw * self.wall_x; acc[2] += alpha * pw * self.wall_y
        acc[6] += (1.0 - alpha) * pw * self.wall_x; acc[7] += (1.0 - alpha) * pw * self.wall_y
        k = dt / self.dx
        hs_n = (hs + k * acc[0]).clamp_min(0.0); hsu_n = self.hsu + k * acc[1]; hsv_n = self.hsv + k * acc[2]
        hi_n = (self.hi + k * acc[3]).clamp_min(0.0); hp_n = (self.hp + k * acc[4]).clamp_min(0.0)
        hf_n = (hf + k * acc[5]).clamp_min(0.0); hfu_n = self.hfu + k * acc[6]; hfv_n = self.hfv + k * acc[7]
        hq_n = (self.hq + k * acc[8]).clamp_min(0.0)
        for arr in (hs_n, hsu_n, hsv_n, hi_n, hp_n, hf_n, hfu_n, hfv_n, hq_n):
            arr[self.edge] = 0.0
        hi_n = torch.minimum(hi_n, hs_n); hp_n = torch.minimum(hp_n, hs_n)
        h_n = hs_n + hf_n
        wet_s = hs_n > self.h_dry; wet_f = hf_n > self.h_dry
        hss = hs_n.clamp_min(self.h_dry); hfs = hf_n.clamp_min(self.h_dry); hts = h_n.clamp_min(self.h_dry)
        us_n = hsu_n / hss; vs_n = hsv_n / hss; uf_n = hfu_n / hfs; vf_n = hfv_n / hfs
        sp_s = torch.sqrt(us_n * us_n + vs_n * vs_n); sp_f = torch.sqrt(uf_n * uf_n + vf_n * vf_n)
        # solid: Coulomb on submerged weight reduced by excess pore pressure
        p = (hp_n / hss).clamp(0.0, 1.0)
        mu_eff = self.mu_s * (1.0 - self.gamma) * (1.0 - p)
        coul = (1.0 - dt * G * mu_eff / sp_s.clamp_min(1e-6)).clamp_min(0.0)
        fac_s = torch.where(wet_s, coul, torch.zeros_like(coul))
        hsu_n = hsu_n * fac_s; hsv_n = hsv_n * fac_s
        # fluid: Manning with the total depth as flow depth
        turb = 1.0 + dt * G * self.n_w * self.n_w * sp_f / hts.pow(4.0 / 3.0)
        fac_f = torch.where(wet_f, 1.0 / turb, torch.zeros_like(turb))
        hfu_n = hfu_n * fac_f; hfv_n = hfv_n * fac_f
        # drag: exact relaxation of the velocity difference, momentum-weighted mean conserved
        both = wet_s & wet_f
        us_n = hsu_n / hss; vs_n = hsv_n / hss; uf_n = hfu_n / hfs; vf_n = hfv_n / hfs
        ms = hss; mf = self.gamma * hfs  # masses relative to rho_s
        lam = (1.0 - self.gamma) * G / self.u_t * (1.0 + hss / (self.gamma * hfs))
        decay = torch.exp(-lam * dt)
        for a_s, a_f, name_s, name_f in ((us_n, uf_n, "hsu", "hfu"), (vs_n, vf_n, "hsv", "hfv")):
            mean = (ms * a_s + mf * a_f) / (ms + mf); w = (a_f - a_s) * decay
            new_s = torch.where(both, mean - mf / (ms + mf) * w, a_s); new_f = torch.where(both, mean + ms / (ms + mf) * w, a_f)
            if name_s == "hsu":
                hsu_n = new_s * hs_n; hfu_n = new_f * hf_n
            else:
                hsv_n = new_s * hs_n; hfv_n = new_f * hf_n
        thin = h_n < 0.02
        for arr in (hsu_n, hsv_n, hfu_n, hfv_n):
            arr[thin] = 0.0
        # a vanishing phase (< 2 cm) moves with the mixture: regularises the internal mode where one phase disappears
        um = (hsu_n + hfu_n) / hts; vm = (hsv_n + hfv_n) / hts
        ts = hs_n < 0.02; tf = hf_n < 0.02
        hsu_n = torch.where(ts, hs_n * um, hsu_n); hsv_n = torch.where(ts, hs_n * vm, hsv_n)
        hfu_n = torch.where(tf, hf_n * um, hfu_n); hfv_n = torch.where(tf, hf_n * vm, hfv_n)
        # heat from friction and drag work, melt solid -> fluid
        if self.melt_eff > 0 or self.melt_tau > 0:
            sp_s = torch.sqrt(hsu_n * hsu_n + hsv_n * hsv_n) / hss; sp_f = torch.sqrt(hfu_n * hfu_n + hfv_n * hfv_n) / hfs
            power = 1000.0 * (2.65 * (1.0 - self.gamma) * mu_eff * G * hs_n * sp_s + G * self.n_w ** 2 * sp_f ** 3 / hts.pow(1.0 / 3.0) * hf_n)
            if self.melt_eff > 0:
                hq_n = hq_n + self.melt_eff * power * dt * wet_f
            dm = torch.minimum(hi_n, hq_n / LAT)
            if self.melt_tau > 0:
                dm = torch.maximum(dm, hi_n * (1.0 - math.exp(-dt / self.melt_tau)))
            dm = torch.minimum(dm, hi_n) * wet_s
            # melt water inherits the solid velocity; pore tracer scaled with the solid mass
            frac = torch.where(hs_n > 0, (hs_n - dm) / hss, torch.zeros_like(hs_n))
            hq_n = (hq_n - dm * LAT).clamp_min(0.0); hi_n = hi_n - dm
            hfu_n = hfu_n + dm * hsu_n / hss; hfv_n = hfv_n + dm * hsv_n / hss
            hsu_n = hsu_n * frac; hsv_n = hsv_n * frac; hp_n = hp_n * frac
            hf_n = hf_n + dm; hs_n = (hs_n - dm).clamp_min(0.0)
            self.melted += float(dm.sum()) * self.dx * self.dx
        if self.erosion_k > 0:
            sp = torch.sqrt(hsu_n * hsu_n + hsv_n * hsv_n) / hss
            cn = hs_n / (hs_n + hf_n).clamp_min(self.h_dry)
            rate = self.erosion_k * (sp - self.erosion_uc).clamp_min(0.0) * wet_s * (cn > 0.05)
            de = torch.minimum(rate * dt, self.erodible)
            self.erodible = self.erodible - de; self.z = self.z - de
            hs_n = hs_n + de * self.erosion_c; hf_n = hf_n + de * (1.0 - self.erosion_c)
            hp_n = hp_n + de * self.erosion_c * self.pore_ent
            self.entrained += float(de.sum()) * self.dx * self.dx
        if self.pore_d > 0:
            hp_n = hp_n * torch.exp(-dt * self.pore_d / (hs_n + hf_n).clamp_min(0.05) ** 2)
        self.hs, self.hsu, self.hsv, self.hi, self.hp = hs_n, hsu_n, hsv_n, hi_n, hp_n
        self.hf, self.hfu, self.hfv, self.hq = hf_n, hfu_n, hfv_n, hq_n
        self.time += dt

    def add_volume(self, idx, volumes_m3, c_value: float = 0.0) -> None:
        dh = volumes_m3 / (self.dx * self.dx)
        self.hf[idx] += dh; self.hq[idx] += dh * 4.2e6 * self.river_temp_c

    def add_release(self, idx, thick: float, c0: float, ice_fraction: float, p0: float) -> None:
        self.hs[idx] += thick * c0; self.hf[idx] += thick * (1.0 - c0)
        self.hi[idx] += thick * ice_fraction; self.hp[idx] += thick * c0 * p0


def run(args: argparse.Namespace) -> None:
    inp = np.load(args.inputs)
    meta = json.loads(Path(args.inputs).with_suffix(".json").read_text("utf-8"))
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    dev = args.device
    active = inp["active"]
    model = TwoPhaseSWE(inp["z"], active, meta["cell_m"], device=dev, mu_s=args.mu_s, n_w=args.n_w, n_d=args.n_d,
                        gamma=args.gamma, u_t=args.u_t, pore_d=args.pore_d, pore_ent=args.pore_ent, cfl=args.cfl)
    print(f"two-phase: active cells {model.n} of {active.size} ({model.n/active.size*100:.1f}%) gamma={args.gamma} U_T={args.u_t} "
          f"mu={args.mu_s} p0={args.pore_p0} D={args.pore_d}", flush=True)
    dx2 = model.dx ** 2
    lat2d = inp["lateral_q_active_m3_s"]
    lr, lc = np.nonzero(lat2d > 0)
    lat_idx, ok = model.to_index(lr, lc); lat_q = torch.as_tensor(lat2d[lr, lc][ok] * args.inflow_scale, dtype=torch.float32, device=dev)
    ext_idx, ok = model.to_index([i["row"] for i in meta["external_inflows"]], [i["col"] for i in meta["external_inflows"]])
    ext_q = torch.tensor([i["q_m3_s"] for i in meta["external_inflows"]], dtype=torch.float32, device=dev)[torch.as_tensor(np.nonzero(ok)[0], device=dev)] * args.inflow_scale
    T = lambda a: torch.as_tensor(a.ravel()[model.flat], device=dev)
    route_chain = T(inp["route_chainage_m"]); upper_chain = T(inp["upper_chainage_m"])
    transect_path = Path(args.inputs).with_name(Path(args.inputs).stem + "_transects.json")
    transects = json.loads(transect_path.read_text("utf-8")) if transect_path.exists() else {}
    for tr in transects.values():
        tr["idx_t"], _ = model.to_index(tr["rows"], tr["cols"])
    model.melt_eff, model.melt_tau, model.river_temp_c = args.melt_eff, args.melt_tau, args.river_temp_c
    if args.erosion_k > 0:
        model.erosion_k, model.erosion_uc, model.erosion_c = args.erosion_k, args.erosion_uc, args.erosion_c
        model.erodible = T(np.where(inp["channel"], args.erodible_depth, 0.0).astype(np.float32))
        print(f"entrainment on: K={args.erosion_k} u_c={args.erosion_uc} depth={args.erodible_depth} m on channel cells")
    if args.restart:
        st = np.load(args.restart)
        h0 = T(st["h"].astype(np.float32)); hc0 = T(st["hc"].astype(np.float32)) if "hc" in st else torch.zeros_like(h0)
        u0 = torch.where(h0 > 1e-3, T(st["hu"].astype(np.float32)) / h0.clamp_min(1e-3), torch.zeros_like(h0))
        v0 = torch.where(h0 > 1e-3, T(st["hv"].astype(np.float32)) / h0.clamp_min(1e-3), torch.zeros_like(h0))
        model.hs = hc0.clamp(0.0); model.hf = (h0 - hc0).clamp_min(0.0)
        model.hsu = model.hs * u0; model.hsv = model.hs * v0; model.hfu = model.hf * u0; model.hfv = model.hf * v0
        if "hi" in st:
            model.hi = torch.minimum(T(st["hi"].astype(np.float32)), model.hs)
        model.hq = model.hf * 4.2e6 * args.river_temp_c
        print(f"restarted from {args.restart}")
    elif args.init_normal_depth:
        acc = T(inp["acc_cells"].astype(np.float32)); channel = T(inp["channel"])
        q = meta["uniform_runoff_m_s"] * acc * dx2 * args.inflow_scale
        zx = model.z[model._nbr_r] - model.z[model._nbr_l]; zy = model.z[model._nbr_d] - model.z[model._nbr_u]
        slope = torch.sqrt(zx * zx + zy * zy).div(2 * model.dx).clamp_min(1e-4)
        hn = (args.n_w * q / (model.dx * torch.sqrt(slope))) ** 0.6
        model.hf = torch.where(channel, hn, torch.zeros_like(hn))
        model.hq = model.hf * 4.2e6 * args.river_temp_c
        print(f"normal-depth init: channel volume {float(model.hf.sum()) * dx2 / 1e6:.2f} Mm3")
    model.time = 0.0
    h_base = model.h.clone()
    release_idx = None
    if args.release:
        release_idx, _ = model.to_index(*np.nonzero(inp["source_mask"]))
        thick = float(inp["source_thickness_m"]) * args.release_volume_scale
        if args.release_duration <= 0:
            model.add_release(release_idx, thick, args.release_c0, args.release_ice_fraction, args.pore_p0)
        release_rate = thick / max(args.release_duration, 1e-9)
        print(f"release {len(release_idx) * dx2 * thick / 1e6:.2f} Mm3 over {len(release_idx)} cells, solid c0 {args.release_c0}, ice {args.release_ice_fraction}, "
              f"pore p0 {args.pore_p0}, duration {args.release_duration} s")

    fields = ["time_s", "dt_s", "wall_s", "total_volume_m3", "debris_volume_m3", "entrained_m3", "ice_volume_m3", "melted_m3",
              "max_h_m", "max_speed_m_s", "debris_front_route_km", "debris_front_upper_km", "flood_front_route_km", "max_solid_speed_m_s", "mean_pore_ratio"]
    for key in transects:
        fields += [f"{key}_Q", f"{key}_Qdebris", f"{key}_hmax", f"{key}_stage", f"{key}_cmax", f"{key}_wet_width_m"]
    wf = (out / "series.csv").open("w", newline="", encoding="utf-8"); writer = csv.DictWriter(wf, fieldnames=fields); writer.writeheader()

    def front(chain, mask):
        vals = chain[mask]; vals = vals[~torch.isnan(vals)]
        return float(vals.max()) / 1e3 if vals.numel() else float("nan")

    t_end, frame_dt = args.t_end, args.frame_dt
    next_frame, frame_no, step_no, injected = 0.0, 0, 0, 0.0
    wall0 = time.time()
    while model.time < t_end - 1e-9:
        dt = min(model.max_dt(), t_end - model.time, next_frame + 1e-9 - model.time if next_frame > model.time else frame_dt)
        dt = max(dt, 1e-3)
        model.add_volume(lat_idx, lat_q * dt); model.add_volume(ext_idx, ext_q * dt)
        injected += float((lat_q.sum() + ext_q.sum()) * dt)
        if release_idx is not None and args.release_duration > 0 and model.time < args.release_duration:
            d_th = release_rate * min(dt, args.release_duration - model.time)
            model.add_release(release_idx, d_th, args.release_c0, args.release_ice_fraction, args.pore_p0)
        model.step(dt); step_no += 1
        if model.time >= next_frame - 1e-6:
            h = model.h; hu = model.hu; hv = model.hv
            inv = torch.where(h > 1e-3, 1.0 / h.clamp_min(1e-3), torch.zeros_like(h))
            c = (model.hs * inv).clamp(0.0, 1.0); speed = torch.sqrt(hu * hu + hv * hv) * inv
            us, vs, _, _ = model.phase_velocities(); sp_s = torch.sqrt(us * us + vs * vs)
            row = {"time_s": round(model.time, 3), "dt_s": dt, "wall_s": round(time.time() - wall0, 1),
                   "total_volume_m3": float(h.sum()) * dx2, "debris_volume_m3": float(model.hs.sum()) * dx2,
                   "entrained_m3": model.entrained, "ice_volume_m3": float(model.hi.sum()) * dx2, "melted_m3": model.melted,
                   "max_h_m": float(h.max()), "max_speed_m_s": float(speed.max()), "max_solid_speed_m_s": float(sp_s.max()),
                   "mean_pore_ratio": float(model.hp.sum() / model.hs.sum().clamp_min(1e-6)),
                   "debris_front_route_km": front(route_chain, (h > 0.1) & (c > 0.05)),
                   "debris_front_upper_km": front(upper_chain, (h > 0.1) & (c > 0.05)),
                   "flood_front_route_km": front(route_chain, (h - h_base) > 0.5)}
            for key, tr in transects.items():
                i = tr["idx_t"]; tx, ty = tr["tangent"]; fs = model.dx * tr.get("flux_scale", 1.0)
                flux = (hu[i] * tx + hv[i] * ty) * fs; flux_s = (model.hsu[i] * tx + model.hsv[i] * ty) * fs
                ht = h[i]; ct = c[i]; wet = ht > 0.05
                row[f"{key}_Q"] = float(flux.sum()); row[f"{key}_Qdebris"] = float(flux_s.sum())
                row[f"{key}_hmax"] = float(ht.max()); row[f"{key}_stage"] = float((model.z[i] + ht)[wet].min()) if bool(wet.any()) else float("nan")
                row[f"{key}_cmax"] = float(ct[wet].max()) if bool(wet.any()) else 0.0; row[f"{key}_wet_width_m"] = float(wet.sum()) * model.dx
            writer.writerow(row); wf.flush()
            if args.save_frames:
                sel = torch.nonzero(h > 0.05).ravel()
                np.savez_compressed(out / "frames" / f"frame_{frame_no:05d}.npz", time_s=model.time,
                                    idx=model.flat[sel.cpu().numpy()].astype(np.int32),
                                    h=h[sel].cpu().numpy().astype(np.float16), c=c[sel].cpu().numpy().astype(np.float16),
                                    ice=(model.hi / h.clamp_min(1e-3))[sel].cpu().numpy().astype(np.float16),
                                    speed=speed[sel].cpu().numpy().astype(np.float16),
                                    hu=hu[sel].cpu().numpy().astype(np.float32), hv=hv[sel].cpu().numpy().astype(np.float32),
                                    us=sp_s[sel].cpu().numpy().astype(np.float16),
                                    pore=(model.hp / model.hs.clamp_min(1e-3))[sel].cpu().numpy().astype(np.float16))
            print(f"t={model.time:8.1f}s step={step_no} dt={dt:.3f} wall={row['wall_s']:.0f}s V={row['total_volume_m3']/1e6:.2f}Mm3 "
                  f"hmax={row['max_h_m']:.1f} umax={row['max_speed_m_s']:.1f} us={row['max_solid_speed_m_s']:.1f} p={row['mean_pore_ratio']:.2f} "
                  f"debris_front={row['debris_front_route_km']:.1f}km flood_front={row['flood_front_route_km']:.1f}km", flush=True)
            frame_no += 1; next_frame += frame_dt
    wf.close()
    np.savez_compressed(out / "final_state.npz", h=model.full(model.h), hu=model.full(model.hu), hv=model.full(model.hv),
                        hc=model.full(model.hs), hi=model.full(model.hi), hq=model.full(model.hq), hp=model.full(model.hp),
                        hsu=model.full(model.hsu), hsv=model.full(model.hsv), time_s=model.time, bed_change=model.full(model.z - model.z0))
    result = {"status": "ok", "steps": step_no, "wall_s": time.time() - wall0, "t_end": model.time, "injected_m3": injected,
              "entrained_m3": model.entrained, "melted_m3": model.melted, "final_volume_m3": float(model.h.sum()) * dx2,
              "final_debris_m3": float(model.hs.sum()) * dx2, "active_cells": model.n, "args": vars(args), "inputs_meta": meta}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("done", json.dumps({k: result[k] for k in ("steps", "wall_s", "final_volume_m3")}))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--restart"); p.add_argument("--release", action="store_true"); p.add_argument("--init-normal-depth", action="store_true")
    p.add_argument("--release-c0", type=float, default=0.9, help="solid volume fraction of the release")
    p.add_argument("--release-volume-scale", type=float, default=1.0)
    p.add_argument("--release-ice-fraction", type=float, default=0.25); p.add_argument("--release-duration", type=float, default=30.0)
    p.add_argument("--melt-eff", type=float, default=1.0); p.add_argument("--melt-tau", type=float, default=0.0)
    p.add_argument("--river-temp-c", type=float, default=12.0)
    p.add_argument("--erosion-k", type=float, default=0.0); p.add_argument("--erosion-uc", type=float, default=3.0)
    p.add_argument("--erosion-c", type=float, default=0.7); p.add_argument("--erodible-depth", type=float, default=8.0)
    p.add_argument("--t-end", type=float, required=True); p.add_argument("--frame-dt", type=float, default=60.0)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--mu-s", type=float, default=0.4, help="dry Coulomb friction of the solid phase")
    p.add_argument("--n-w", type=float, default=0.03); p.add_argument("--n-d", type=float, default=0.03)
    p.add_argument("--gamma", type=float, default=0.4, help="rho_f / rho_s"); p.add_argument("--u-t", type=float, default=1.0, help="drag terminal velocity (m/s)")
    p.add_argument("--pore-p0", type=float, default=1.0, help="initial excess pore-pressure ratio of the release")
    p.add_argument("--pore-d", type=float, default=0.0, help="pore-pressure diffusivity (m^2/s); 0 = no dissipation")
    p.add_argument("--pore-ent", type=float, default=1.0, help="pore-pressure ratio of entrained bed material")
    p.add_argument("--cfl", type=float, default=0.45); p.add_argument("--inflow-scale", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
