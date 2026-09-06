"""Active-cell (unstructured 4-neighbour) version of the mixture shallow-water solver.

Same physics and numerics as solver_swe.py (HLL, depth-limited in-cell bed tilt with
hydrostatic reconstruction, Coulomb+Manning friction, ice/heat tracers, melt, entrainment),
but the state lives on a 1-D array of active cells and fluxes are evaluated on explicit
face lists, so only the corridor is computed.
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

G = 9.81
LAT = 900.0 * 3.34e5


class SparseSWE:
    def __init__(self, z2d: np.ndarray, active: np.ndarray, cell: float, *, device: str, mu_s: float, n_w: float,
                 n_d: float, h_dry: float = 1e-3, cfl: float = 0.45) -> None:
        self.dev = torch.device(device)
        self.dx = float(cell)
        self.mu_s, self.n_w, self.n_d, self.h_dry, self.cfl = mu_s, n_w, n_d, h_dry, cfl
        self.shape = z2d.shape
        rows, cols = z2d.shape
        self.aidx = np.full(z2d.shape, -1, np.int64)
        flat = np.nonzero(active.ravel())[0]
        self.flat = flat
        self.aidx.ravel()[flat] = np.arange(len(flat))
        self.n = len(flat)
        r, c = np.divmod(flat, cols)
        z = z2d.ravel()[flat].astype(np.float32)

        def nb(dr, dc):
            rr, cc = r + dr, c + dc
            ok = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
            out = np.full(self.n, -1, np.int64)
            out[ok] = self.aidx[rr[ok], cc[ok]]
            return out
        self.right, self.left, self.down, self.up = nb(0, 1), nb(0, -1), nb(1, 0), nb(-1, 0)
        T = lambda a: torch.as_tensor(a, device=self.dev)
        # faces: x between i and right(i); y between i (upper row) and down(i)
        fx = np.nonzero(self.right >= 0)[0]; self.fxL = T(fx); self.fxR = T(self.right[fx])
        fy = np.nonzero(self.down >= 0)[0]; self.fyL = T(fy); self.fyR = T(self.down[fy])
        self.z = T(z)
        zt = self.z
        def zof(nbr):  # neighbour bed or own bed when missing
            idx = torch.as_tensor(np.where(nbr >= 0, nbr, np.arange(self.n)), device=self.dev)
            return zt[idx]
        self.zx_plus0 = 0.5 * (zt + zof(self.right)); self.zx_minus0 = 0.5 * (zt + zof(self.left))
        self.zy_plus0 = 0.5 * (zt + zof(self.down)); self.zy_minus0 = 0.5 * (zt + zof(self.up))
        self._nbr_r = torch.as_tensor(np.where(self.right >= 0, self.right, np.arange(self.n)), device=self.dev)
        self._nbr_l = torch.as_tensor(np.where(self.left >= 0, self.left, np.arange(self.n)), device=self.dev)
        self._nbr_d = torch.as_tensor(np.where(self.down >= 0, self.down, np.arange(self.n)), device=self.dev)
        self._nbr_u = torch.as_tensor(np.where(self.up >= 0, self.up, np.arange(self.n)), device=self.dev)
        edge = (r == 0) | (r == rows - 1) | (c == 0) | (c == cols - 1)
        # cells with a missing neighbour along an axis get no in-cell bed tilt on that axis (wall)
        self.tilt_ok_x = T(((self.right >= 0) & (self.left >= 0)).astype(np.float32))
        self.wall_x = T(((self.left < 0).astype(np.float32) - (self.right < 0).astype(np.float32)))  # +1: missing left, -1: missing right
        self.wall_y = T(((self.down < 0).astype(np.float32) - (self.up < 0).astype(np.float32)))  # +1: missing down, -1: missing up
        self.tilt_ok_y = T(((self.down >= 0) & (self.up >= 0)).astype(np.float32))
        self.edge = T(np.nonzero(edge)[0])
        zero = lambda: torch.zeros(self.n, dtype=torch.float32, device=self.dev)
        self.h, self.hu, self.hv, self.hc, self.hi, self.hq = zero(), zero(), zero(), zero(), zero(), zero()
        self.z0 = self.z.clone()
        self.erodible = zero()
        self.erosion_k, self.erosion_uc, self.erosion_c = 0.0, 0.0, 0.7
        self.melt_eff, self.melt_tau, self.river_temp_c = 0.0, 0.0, 0.0
        self.entrained = 0.0; self.melted = 0.0; self.time = 0.0
        self.eta0, self.eta_beta = 0.0, 0.0
        self.mu_dry, self.w_sat = 0.0, 0.36
        self.q_air = 0.0  # W/m^2 to the wet surface (air + sun)
        self.melt_energy = False  # credit the whole mechanical-energy loss per step (friction + shocks) to the heat store
        inner = np.ones(self.n, bool); inner[np.nonzero(edge)[0]] = False; self.inner = torch.as_tensor(inner, device=self.dev)
        self.dep_uc, self.dep_tau, self.deposited = 0.0, 600.0, 0.0
        self.dissipated = 0.0

    def to_index(self, rows, cols):
        a = self.aidx[np.asarray(rows), np.asarray(cols)]
        return torch.as_tensor(a[a >= 0], device=self.dev), a >= 0

    def full(self, x: torch.Tensor) -> np.ndarray:
        out = np.zeros(self.shape, np.float32).ravel(); out[self.flat] = x.cpu().numpy(); return out.reshape(self.shape)

    def velocities(self):
        wet = self.h > self.h_dry
        inv = torch.where(wet, 1.0 / self.h.clamp_min(self.h_dry), torch.zeros_like(self.h))
        return self.hu * inv, self.hv * inv, (self.hc * inv).clamp(0.0, 1.0), wet

    def max_dt(self) -> float:
        u, v, _, _ = self.velocities()
        s = torch.maximum(u.abs(), v.abs()) + torch.sqrt(2.0 * G * self.h.clamp_min(0.0))
        smax = float(s.max())
        return 5.0 if smax <= 1e-6 else min(self.cfl * self.dx / smax, 5.0)

    @staticmethod
    def _hll(hL, hR, uL, uR):
        cL_ = torch.sqrt(G * hL); cR_ = torch.sqrt(G * hR)
        sL = torch.minimum(uL - cL_, uR - cR_); sR = torch.maximum(uL + cL_, uR + cR_)
        fL0, fR0 = hL * uL, hR * uR
        fL1 = hL * uL * uL + 0.5 * G * hL * hL; fR1 = hR * uR * uR + 0.5 * G * hR * hR
        denom = sR - sL; safe = denom.abs() > 1e-8
        w = torch.where(safe, 1.0 / torch.where(safe, denom, torch.ones_like(denom)), torch.zeros_like(denom))
        f0 = (sR * fL0 - sL * fR0 + sL * sR * (hR - hL)) * w
        f1 = (sR * fL1 - sL * fR1 + sL * sR * (hR * uR - hL * uL)) * w
        f0 = torch.where(sL >= 0, fL0, torch.where(sR <= 0, fR0, f0))
        f1 = torch.where(sL >= 0, fL1, torch.where(sR <= 0, fR1, f1))
        return f0, f1, f0 >= 0

    def _axis(self, h, un, ut, c, ice, heat, zplus, zminus, L, R, tilt_ok):
        """Face fluxes along one axis. un = normal velocity (+ toward increasing index)."""
        tilt = torch.clamp(0.5 * (zplus - zminus), -h, h) * tilt_ok
        zP = self.z + tilt; zM = self.z - tilt; hE = h - tilt; hW = h + tilt
        zf = torch.maximum(zP[L], zM[R])
        hL = (hE[L] + zP[L] - zf).clamp_min(0.0); hR = (hW[R] + zM[R] - zf).clamp_min(0.0)
        f0, f1, up = self._hll(hL, hR, un[L], un[R])
        pick = lambda a: torch.where(up, a[L], a[R])
        f2 = f0 * pick(ut); f3 = f0 * pick(c); f4 = f0 * pick(ice); f5 = f0 * pick(heat)
        f1l = f1 + 0.5 * G * (hE[L] ** 2 - hL ** 2); f1r = f1 + 0.5 * G * (hW[R] ** 2 - hR ** 2)
        src = 0.5 * G * (hE + hW) * (zM - zP)
        return f0, f1l, f1r, f2, f3, f4, f5, src

    def step(self, dt: float) -> None:
        h, hu, hv, hc, hi, hq = self.h, self.hu, self.hv, self.hc, self.hi, self.hq
        u, v, c, wet0 = self.velocities()
        inv = torch.where(wet0, 1.0 / h.clamp_min(self.h_dry), torch.zeros_like(h))
        ice = (hi * inv).clamp(0.0, 1.0); heat = hq * inv
        z = self.z
        if self.melt_energy:
            rho0 = 1000.0 * (1.0 + 1.65 * c)
            E0 = (rho0 * (0.5 * (hu * hu + hv * hv) * inv + G * h * (z + 0.5 * h)) * self.inner).sum()
        # bed at faces follows the current (possibly eroded) bed
        zx_plus = 0.5 * (z + z[self._nbr_r]); zx_minus = 0.5 * (z + z[self._nbr_l])
        zy_plus = 0.5 * (z + z[self._nbr_d]); zy_minus = 0.5 * (z + z[self._nbr_u])
        acc = [torch.zeros_like(h) for _ in range(6)]  # dh, dhu, dhv, dhc, dhi, dhq
        # x faces
        f0, f1l, f1r, f2, f3, f4, f5, src = self._axis(h, u, v, c, ice, heat, zx_plus, zx_minus, self.fxL, self.fxR, self.tilt_ok_x)
        L, R = self.fxL, self.fxR
        acc[0].index_add_(0, L, -f0); acc[0].index_add_(0, R, f0)
        acc[1].index_add_(0, L, -f1l); acc[1].index_add_(0, R, f1r); acc[1] += src
        acc[2].index_add_(0, L, -f2); acc[2].index_add_(0, R, f2)
        acc[3].index_add_(0, L, -f3); acc[3].index_add_(0, R, f3)
        acc[4].index_add_(0, L, -f4); acc[4].index_add_(0, R, f4)
        acc[5].index_add_(0, L, -f5); acc[5].index_add_(0, R, f5)
        # y faces: increasing row = south; normal velocity is -v
        f0, f1l, f1r, f2, f3, f4, f5, src = self._axis(h, -v, u, c, ice, heat, zy_plus, zy_minus, self.fyL, self.fyR, self.tilt_ok_y)
        L, R = self.fyL, self.fyR
        acc[0].index_add_(0, L, -f0); acc[0].index_add_(0, R, f0)
        acc[2].index_add_(0, L, f1l); acc[2].index_add_(0, R, -f1r); acc[2] -= src
        acc[1].index_add_(0, L, -f2); acc[1].index_add_(0, R, f2)
        acc[3].index_add_(0, L, -f3); acc[3].index_add_(0, R, f3)
        acc[4].index_add_(0, L, -f4); acc[4].index_add_(0, R, f4)
        acc[5].index_add_(0, L, -f5); acc[5].index_add_(0, R, f5)
        # missing neighbours act as walls: their hydrostatic pressure 0.5*g*h^2 on the cell
        acc[1] += 0.5 * G * h * h * self.wall_x
        acc[2] += 0.5 * G * h * h * self.wall_y
        k = dt / self.dx
        h_new = (h + k * acc[0]).clamp_min(0.0); hu_new = hu + k * acc[1]; hv_new = hv + k * acc[2]
        hc_new = (hc + k * acc[3]).clamp_min(0.0); hi_new = (hi + k * acc[4]).clamp_min(0.0); hq_new = (hq + k * acc[5]).clamp_min(0.0)
        if self.melt_energy:  # energy leaving through the domain edge is not dissipation
            he = h_new[self.edge].clamp_min(self.h_dry); ce = (hc_new[self.edge] / he).clamp(0.0, 1.0)
            e_edge = (1000.0 * (1.0 + 1.65 * ce) * (0.5 * (hu_new[self.edge] ** 2 + hv_new[self.edge] ** 2) / he + G * h_new[self.edge] * (z[self.edge] + 0.5 * h_new[self.edge]))).sum()
        for arr in (h_new, hu_new, hv_new, hc_new, hi_new, hq_new):
            arr[self.edge] = 0.0
        wet = h_new > self.h_dry
        hs = h_new.clamp_min(self.h_dry)
        un = hu_new / hs; vn = hv_new / hs
        cn = (hc_new / hs).clamp(0.0, 1.0)
        speed = torch.sqrt(un * un + vn * vn)
        mu = self.mu_s * cn
        if self.mu_dry > 0:  # dry friction that vanishes as the local water fraction reaches the saturation value w_sat
            mu = mu + self.mu_dry * (1.0 - (1.0 - cn) / self.w_sat).clamp(0.0, 1.0)
        n = self.n_w + (self.n_d - self.n_w) * cn
        turb = 1.0 + dt * G * n * n * speed / hs.pow(4.0 / 3.0)
        if self.eta0 > 0:  # viscous (laminar) resistance of the mixture, K eta u / (8 rho_m h^2), Julien & Lan (1991)
            eta = self.eta0 * torch.exp(self.eta_beta * cn)
            turb = turb + dt * 3.0 * eta / (1000.0 * (1.0 + 1.65 * cn) * hs * hs)
        coul = (1.0 - dt * G * mu / speed.clamp_min(1e-6)).clamp_min(0.0)
        fac = torch.where(wet, coul / turb, torch.zeros_like(h_new))
        hc_new = torch.minimum(hc_new, h_new); hi_new = torch.minimum(hi_new, hc_new)
        hu_new = hu_new * fac; hv_new = hv_new * fac
        thin = h_new < 0.02  # films thinner than 2 cm carry no momentum (robustness, CFL)
        hu_new = torch.where(thin, torch.zeros_like(hu_new), hu_new); hv_new = torch.where(thin, torch.zeros_like(hv_new), hv_new)
        if self.melt_eff > 0 or self.melt_tau > 0:
            sp2 = torch.sqrt(hu_new * hu_new + hv_new * hv_new) / hs
            accel = G * mu + G * n * n * sp2 * sp2 / hs.pow(4.0 / 3.0)
            power = 1000.0 * (1.0 + 0.8 * cn) * h_new * accel * sp2
            if self.melt_energy:
                rho2 = 1000.0 * (1.0 + 1.65 * cn)
                E2 = (rho2 * (0.5 * (hu_new * hu_new + hv_new * hv_new) / hs + G * h_new * (z + 0.5 * h_new)) * self.inner).sum()
                D = (E0 - E2 - e_edge).clamp_min(0.0)  # J/m^2 summed over cells
                w = (h_new * sp2 * wet).clamp_min(0.0); wsum = w.sum().clamp_min(1e-9)
                hq_new = hq_new + self.melt_eff * D * w / wsum
                self.dissipated = self.dissipated + D * self.dx * self.dx
            elif self.melt_eff > 0:
                hq_new = hq_new + self.melt_eff * power * dt * wet
            if self.q_air > 0:
                hq_new = hq_new + self.q_air * dt * wet
            dm = torch.minimum(hi_new, hq_new / LAT)
            if self.melt_tau > 0:
                dm = torch.maximum(dm, hi_new * (1.0 - math.exp(-dt / self.melt_tau)))
            dm = torch.minimum(dm, hi_new) * wet
            hq_new = (hq_new - dm * LAT).clamp_min(0.0); hi_new = hi_new - dm; hc_new = (hc_new - dm).clamp_min(0.0)
            self.melted += float(dm.sum()) * self.dx * self.dx
        if self.erosion_k > 0:
            sp = torch.sqrt(hu_new * hu_new + hv_new * hv_new) / hs
            rate = self.erosion_k * (sp - self.erosion_uc).clamp_min(0.0) * wet * (cn > 0.05)
            de = torch.minimum(rate * dt, self.erodible)
            self.erodible = self.erodible - de; self.z = self.z - de
            h_new = h_new + de; hc_new = hc_new + de * self.erosion_c
            self.entrained += float(de.sum()) * self.dx * self.dx
        if self.dep_uc > 0:  # settling of solids where the flow is slower than dep_uc, timescale dep_tau
            sp = torch.sqrt(hu_new * hu_new + hv_new * hv_new) / hs
            cn2 = (hc_new / hs).clamp(0.0, 1.0)
            rate = (1.0 - sp / self.dep_uc).clamp_min(0.0) * wet * (cn2 > 0.05) * (1.0 - math.exp(-dt / self.dep_tau))
            dd = torch.minimum(hc_new - hi_new, hc_new * rate).clamp_min(0.0)  # ice does not settle
            self.z = self.z + dd; h_new = (h_new - dd).clamp_min(0.0); hc_new = (hc_new - dd).clamp_min(0.0)
            self.deposited += float(dd.sum()) * self.dx * self.dx
        self.h, self.hu, self.hv, self.hc, self.hi, self.hq = h_new, hu_new, hv_new, hc_new, hi_new, hq_new
        self.time += dt

    def add_volume(self, idx, volumes_m3, c_value: float = 0.0) -> None:
        dh = volumes_m3 / (self.dx * self.dx)
        self.h[idx] += dh; self.hc[idx] += dh * c_value; self.hq[idx] += dh * 4.2e6 * self.river_temp_c


def run(args: argparse.Namespace) -> None:
    inp = np.load(args.inputs)
    meta = json.loads(Path(args.inputs).with_suffix(".json").read_text("utf-8"))
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    dev = args.device
    active = inp["active"]
    model = SparseSWE(inp["z"], active, meta["cell_m"], device=dev, mu_s=args.mu_s, n_w=args.n_w, n_d=args.n_d, cfl=args.cfl)
    print(f"active cells {model.n} of {active.size} ({model.n/active.size*100:.1f}%)", flush=True)
    dx2 = model.dx ** 2
    lat2d = inp["lateral_q_active_m3_s"]
    lr, lc = np.nonzero(lat2d > 0)
    lat_idx, ok = model.to_index(lr, lc); lat_q = torch.as_tensor(lat2d[lr, lc][ok] * args.inflow_scale, dtype=torch.float32, device=dev)
    ext_idx, ok = model.to_index([i["row"] for i in meta["external_inflows"]], [i["col"] for i in meta["external_inflows"]])
    ext_q = torch.tensor([i["q_m3_s"] for i in meta["external_inflows"]], dtype=torch.float32, device=dev)[torch.as_tensor(np.nonzero(ok)[0], device=dev)] * args.inflow_scale
    T = lambda a: torch.as_tensor(a.ravel()[model.flat], device=dev)
    route_chain = T(inp["route_chainage_m"]); upper_chain = T(inp["upper_chainage_m"])
    stations = meta["stations"]
    transect_path = Path(args.inputs).with_name(Path(args.inputs).stem + "_transects.json")
    transects = json.loads(transect_path.read_text("utf-8")) if transect_path.exists() else {}
    for tr in transects.values():
        tr["idx_t"], _ = model.to_index(tr["rows"], tr["cols"])
    model.melt_eff, model.melt_tau, model.river_temp_c = args.melt_eff, args.melt_tau, args.river_temp_c
    model.eta0, model.eta_beta, model.dep_uc, model.dep_tau = args.eta0, args.eta_beta, args.dep_uc, args.dep_tau
    model.melt_energy = bool(args.melt_energy); model.mu_dry, model.w_sat = args.mu_dry, args.w_sat; model.q_air = args.q_air
    if args.erosion_k > 0:
        model.erosion_k, model.erosion_uc, model.erosion_c = args.erosion_k, args.erosion_uc, args.erosion_c
        model.erodible = T(np.where(inp["channel"], args.erodible_depth, 0.0).astype(np.float32))
        print(f"entrainment on: K={args.erosion_k} u_c={args.erosion_uc} depth={args.erodible_depth} m on channel cells")
    if args.restart:
        st = np.load(args.restart)
        for key in ("h", "hu", "hv", "hc", "hi"):
            if key in st:
                setattr(model, key, T(st[key].astype(np.float32)))
        model.hq = (model.h - model.hc).clamp_min(0.0) * 4.2e6 * args.river_temp_c
        print(f"restarted from {args.restart}")
    elif args.init_normal_depth:
        acc = T(inp["acc_cells"].astype(np.float32)); channel = T(inp["channel"])
        q = meta["uniform_runoff_m_s"] * acc * dx2 * args.inflow_scale
        zx = model.z[model._nbr_r] - model.z[model._nbr_l]; zy = model.z[model._nbr_d] - model.z[model._nbr_u]
        slope = torch.sqrt(zx * zx + zy * zy).div(2 * model.dx).clamp_min(1e-4)
        hn = (args.n_w * q / (model.dx * torch.sqrt(slope))) ** 0.6
        model.h = torch.where(channel, hn, torch.zeros_like(hn))
        model.hq = model.h * 4.2e6 * args.river_temp_c
        print(f"normal-depth init: channel volume {float(model.h.sum()) * dx2 / 1e6:.2f} Mm3")
    model.time = 0.0
    h_base = model.h.clone()
    release_idx = None
    if args.release:
        release_idx, _ = model.to_index(*np.nonzero(inp["source_mask"]))
        thick = float(inp["source_thickness_m"]) * args.release_volume_scale
        rock_heat = 2.1e6 * args.rock_temp_c  # J/m^3 of rock, sensible heat released on cooling to 0 C
        # release velocity: downslope unit vector at each source cell (free-fall impact speed)
        gx = (model.z[model._nbr_r] - model.z[model._nbr_l]) / (2 * model.dx); gy = (model.z[model._nbr_u] - model.z[model._nbr_d]) / (2 * model.dx)
        gn = torch.sqrt(gx * gx + gy * gy).clamp_min(1e-6); rel_ux = (-gx / gn)[release_idx] * args.release_speed; rel_uy = (-gy / gn)[release_idx] * args.release_speed
        if args.release_duration <= 0:
            model.h[release_idx] += thick; model.hc[release_idx] += thick * args.release_c0; model.hi[release_idx] += thick * args.release_ice_fraction
            model.hq[release_idx] += thick * (args.release_c0 - args.release_ice_fraction) * rock_heat
            model.hu[release_idx] += thick * rel_ux; model.hv[release_idx] += thick * rel_uy
        release_rate = thick / max(args.release_duration, 1e-9)
        print(f"release {len(release_idx) * dx2 * thick / 1e6:.2f} Mm3 over {len(release_idx)} cells, c0 {args.release_c0}, ice {args.release_ice_fraction}, duration {args.release_duration} s")

    fields = ["time_s", "dt_s", "wall_s", "total_volume_m3", "debris_volume_m3", "entrained_m3", "ice_volume_m3", "melted_m3",
              "max_h_m", "max_speed_m_s", "debris_front_route_km", "debris_front_upper_km", "flood_front_route_km"]
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
            model.h[release_idx] += d_th; model.hc[release_idx] += d_th * args.release_c0; model.hi[release_idx] += d_th * args.release_ice_fraction
            model.hq[release_idx] += d_th * (args.release_c0 - args.release_ice_fraction) * rock_heat
            model.hu[release_idx] += d_th * rel_ux; model.hv[release_idx] += d_th * rel_uy
        model.step(dt); step_no += 1
        if model.time >= next_frame - 1e-6:
            u, v, c, _ = model.velocities(); speed = torch.sqrt(u * u + v * v)
            row = {"time_s": round(model.time, 3), "dt_s": dt, "wall_s": round(time.time() - wall0, 1),
                   "total_volume_m3": float(model.h.sum()) * dx2, "debris_volume_m3": float(model.hc.sum()) * dx2,
                   "entrained_m3": model.entrained, "ice_volume_m3": float(model.hi.sum()) * dx2, "melted_m3": model.melted,
                   "max_h_m": float(model.h.max()), "max_speed_m_s": float(speed.max()),
                   "debris_front_route_km": front(route_chain, (model.h > 0.1) & (c > 0.05)),
                   "debris_front_upper_km": front(upper_chain, (model.h > 0.1) & (c > 0.05)),
                   "flood_front_route_km": front(route_chain, (model.h - h_base) > 0.5)}
            for key, tr in transects.items():
                i = tr["idx_t"]; tx, ty = tr["tangent"]
                flux = (model.hu[i] * tx + model.hv[i] * ty) * model.dx * tr.get("flux_scale", 1.0)
                ht = model.h[i]; ct = c[i]; wet = ht > 0.05
                row[f"{key}_Q"] = float(flux.sum()); row[f"{key}_Qdebris"] = float((flux * ct).sum())
                row[f"{key}_hmax"] = float(ht.max()); row[f"{key}_stage"] = float((model.z[i] + ht)[wet].min()) if bool(wet.any()) else float("nan")
                row[f"{key}_cmax"] = float(ct[wet].max()) if bool(wet.any()) else 0.0; row[f"{key}_wet_width_m"] = float(wet.sum()) * model.dx
            writer.writerow(row); wf.flush()
            if args.save_frames:
                m = model.h > 0.05
                sel = torch.nonzero(m).ravel()
                np.savez_compressed(out / "frames" / f"frame_{frame_no:05d}.npz", time_s=model.time,
                                    idx=model.flat[sel.cpu().numpy()].astype(np.int32),
                                    h=model.h[sel].cpu().numpy().astype(np.float16), c=c[sel].cpu().numpy().astype(np.float16),
                                    ice=(model.hi / model.h.clamp_min(1e-3))[sel].cpu().numpy().astype(np.float16),
                                    speed=speed[sel].cpu().numpy().astype(np.float16),
                                    hu=model.hu[sel].cpu().numpy().astype(np.float32), hv=model.hv[sel].cpu().numpy().astype(np.float32),
                                    dz_idx=model.flat[torch.nonzero((model.z - model.z0).abs() > 0.05).ravel().cpu().numpy()].astype(np.int32),
                                    dz=(model.z - model.z0)[(model.z - model.z0).abs() > 0.05].cpu().numpy().astype(np.float16))
            print(f"t={model.time:8.1f}s step={step_no} dt={dt:.3f} wall={row['wall_s']:.0f}s V={row['total_volume_m3']/1e6:.2f}Mm3 "
                  f"hmax={row['max_h_m']:.1f} umax={row['max_speed_m_s']:.1f} debris_front={row['debris_front_route_km']:.1f}km "
                  f"flood_front={row['flood_front_route_km']:.1f}km", flush=True)
            frame_no += 1; next_frame += frame_dt
    wf.close()
    np.savez_compressed(out / "final_state.npz", h=model.full(model.h), hu=model.full(model.hu), hv=model.full(model.hv),
                        hc=model.full(model.hc), hi=model.full(model.hi), hq=model.full(model.hq), time_s=model.time,
                        bed_change=model.full(model.z - model.z0))
    result = {"status": "ok", "steps": step_no, "wall_s": time.time() - wall0, "t_end": model.time, "injected_m3": injected,
              "entrained_m3": model.entrained, "melted_m3": model.melted, "deposited_m3": model.deposited, "dissipated_J": float(model.dissipated), "final_volume_m3": float(model.h.sum()) * dx2,
              "final_debris_m3": float(model.hc.sum()) * dx2, "active_cells": model.n, "args": vars(args), "inputs_meta": meta}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("done", json.dumps({k: result[k] for k in ("steps", "wall_s", "final_volume_m3")}))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--restart"); p.add_argument("--release", action="store_true"); p.add_argument("--init-normal-depth", action="store_true")
    p.add_argument("--release-c0", type=float, default=1.0); p.add_argument("--release-volume-scale", type=float, default=1.0)
    p.add_argument("--release-ice-fraction", type=float, default=0.0); p.add_argument("--release-duration", type=float, default=0.0)
    p.add_argument("--melt-eff", type=float, default=0.0); p.add_argument("--melt-tau", type=float, default=0.0)
    p.add_argument("--river-temp-c", type=float, default=12.0)
    p.add_argument("--erosion-k", type=float, default=0.0); p.add_argument("--erosion-uc", type=float, default=3.0)
    p.add_argument("--erosion-c", type=float, default=0.7); p.add_argument("--erodible-depth", type=float, default=8.0)
    p.add_argument("--t-end", type=float, required=True); p.add_argument("--frame-dt", type=float, default=60.0)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--mu-s", type=float, default=0.02); p.add_argument("--n-w", type=float, default=0.03); p.add_argument("--n-d", type=float, default=0.03)
    p.add_argument("--cfl", type=float, default=0.45); p.add_argument("--inflow-scale", type=float, default=1.0)
    p.add_argument("--eta0", type=float, default=0.0, help="mixture viscosity at c=0 (Pa s); eta = eta0 exp(eta_beta c)")
    p.add_argument("--eta-beta", type=float, default=0.0)
    p.add_argument("--dep-uc", type=float, default=0.0, help="solids settle where speed < dep_uc (m/s); 0 = off")
    p.add_argument("--dep-tau", type=float, default=600.0, help="settling timescale (s)")
    p.add_argument("--mu-dry", type=float, default=0.0, help="dry Coulomb friction removed progressively as the water fraction approaches w_sat")
    p.add_argument("--w-sat", type=float, default=0.36, help="water volume fraction at which the dry friction vanishes")
    p.add_argument("--release-speed", type=float, default=0.0, help="initial downslope speed of the released material (m/s), e.g. the free-fall impact speed")
    p.add_argument("--rock-temp-c", type=float, default=0.0, help="initial rock temperature; its sensible heat (2.1e6 J/m3/K) goes to the heat store")
    p.add_argument("--q-air", type=float, default=0.0, help="heat flux from air and sun to the wet surface (W/m2)")
    p.add_argument("--melt-energy", action="store_true", help="heat store receives the whole mechanical-energy loss per step (friction and shocks), not only the friction work")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
