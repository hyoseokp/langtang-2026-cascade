"""Single-layer mixture shallow-water solver (PyTorch, first-order Godunov/HLL).

Conserved variables: h, hu, hv, hc where c is the volumetric debris fraction
advected with the flow. Bed friction blends Coulomb (mu*c) with a Manning term
whose roughness rises with c. Hydrostatic reconstruction (Audusse et al. 2004)
keeps a lake at rest exactly balanced and depths non-negative at CFL <= 0.5.
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


class MixtureSWE:
    def __init__(self, z: np.ndarray, cell: float, *, device: str, mu_s: float, n_w: float,
                 n_d: float, h_dry: float = 1e-3, cfl: float = 0.45) -> None:
        self.dev = torch.device(device)
        self.dx = float(cell)
        self.mu_s, self.n_w, self.n_d, self.h_dry, self.cfl = mu_s, n_w, n_d, h_dry, cfl
        self.z = torch.as_tensor(z, dtype=torch.float32, device=self.dev)
        self.h = torch.zeros_like(self.z)
        self.hu = torch.zeros_like(self.z)
        self.hv = torch.zeros_like(self.z)
        self.hc = torch.zeros_like(self.z)
        self.hi = torch.zeros_like(self.z)
        self.hq = torch.zeros_like(self.z)  # sensible heat of liquid water, J/m2
        self.river_temp_c = 0.0
        self.melted = 0.0
        self.melt_eff, self.melt_tau = 0.0, 0.0
        self.time = 0.0
        self.z0 = self.z.clone()
        self.erosion_k, self.erosion_uc, self.erosion_c = 0.0, 0.0, 0.7
        self.erodible = torch.zeros_like(self.z)
        self.entrained = 0.0

    # --- helpers -----------------------------------------------------------
    def velocities(self):
        wet = self.h > self.h_dry
        inv = torch.where(wet, 1.0 / self.h.clamp_min(self.h_dry), torch.zeros_like(self.h))
        u = self.hu * inv
        v = self.hv * inv
        c = (self.hc * inv).clamp(0.0, 1.0)
        return u, v, c, wet

    def max_dt(self) -> float:
        u, v, _, _ = self.velocities()
        cel = torch.sqrt(2.0 * G * self.h.clamp_min(0.0))  # face columns may reach 2h
        s = torch.maximum(u.abs(), v.abs()) + cel
        smax = float(s.max())
        if smax <= 1e-6:
            return 5.0
        return min(self.cfl * self.dx / smax, 5.0)

    @staticmethod
    def _hll(hL, hR, uL, uR, vL, vR, cL, cR):
        """HLL flux for the normal direction; tangential momentum and tracer upwinded."""
        cL_ = torch.sqrt(G * hL)
        cR_ = torch.sqrt(G * hR)
        sL = torch.minimum(uL - cL_, uR - cR_)
        sR = torch.maximum(uL + cL_, uR + cR_)
        fL0, fR0 = hL * uL, hR * uR
        fL1 = hL * uL * uL + 0.5 * G * hL * hL
        fR1 = hR * uR * uR + 0.5 * G * hR * hR
        denom = (sR - sL)
        safe = denom.abs() > 1e-8
        denom = torch.where(safe, denom, torch.ones_like(denom))
        w = torch.where(safe, 1.0 / denom, torch.zeros_like(denom))
        f0 = (sR * fL0 - sL * fR0 + sL * sR * (hR - hL)) * w
        f1 = (sR * fL1 - sL * fR1 + sL * sR * (hR * uR - hL * uL)) * w
        left = sL >= 0
        right = sR <= 0
        f0 = torch.where(left, fL0, torch.where(right, fR0, f0))
        f1 = torch.where(left, fL1, torch.where(right, fR1, f1))
        up = f0 >= 0
        f2 = f0 * torch.where(up, vL, vR)
        f3 = f0 * torch.where(up, cL, cR)
        return f0, f1, f2, f3, up

    def _direction_flux(self, h, u, v, c, z, axis: int):
        """Fluxes across interior faces along `axis` with a continuous piecewise-linear bed.

        Water surface is piecewise constant per cell, the bed is linear between cell
        centres, so face depths follow the bed (hE toward +index, hW toward -index).
        Cells whose face depth would turn negative fall back to a flat depth, which
        keeps the mass flux bounded by the cell content. Returns face fluxes and the
        centred bed-slope momentum source per cell.
        """
        n = h.shape[axis]
        sl_l = [slice(None)] * 2
        sl_r = [slice(None)] * 2
        sl_l[axis] = slice(0, -1)
        sl_r[axis] = slice(1, None)
        zf = 0.5 * (z[sl_l] + z[sl_r])
        pad_first = [slice(None)] * 2; pad_first[axis] = slice(0, 1)
        pad_last = [slice(None)] * 2; pad_last[axis] = slice(n - 1, n)
        z_plus = torch.cat([zf, z[pad_last]], dim=axis)
        z_minus = torch.cat([z[pad_first], zf], dim=axis)
        # bed tilt inside the cell, limited by the local depth so a cell never claims a
        # bed above its own free surface (walls) nor a face column deeper than 2h
        tilt = torch.clamp(0.5 * (z_plus - z_minus), -h, h)
        zP = z + tilt
        zM = z - tilt
        hE = h - tilt
        hW = h + tilt
        # hydrostatic reconstruction at the face between cell i (E side) and i+1 (W side)
        zf_face = torch.maximum(zP[sl_l], zM[sl_r])
        hL = (hE[sl_l] + zP[sl_l] - zf_face).clamp_min(0.0)
        hR = (hW[sl_r] + zM[sl_r] - zf_face).clamp_min(0.0)
        if axis == 1:
            un_L, un_R, ut_L, ut_R = u[sl_l], u[sl_r], v[sl_l], v[sl_r]
        else:
            un_L, un_R, ut_L, ut_R = -v[sl_l], -v[sl_r], u[sl_l], u[sl_r]
        f0, f1, f2, f3, up = self._hll(hL, hR, un_L, un_R, ut_L, ut_R, c[sl_l], c[sl_r])
        f4 = f0 * torch.where(up, self._ice[sl_l], self._ice[sl_r])
        f5 = f0 * torch.where(up, self._heat[sl_l], self._heat[sl_r])
        corr_l = 0.5 * G * (hE[sl_l] ** 2 - hL ** 2)
        corr_r = 0.5 * G * (hW[sl_r] ** 2 - hR ** 2)
        source = 0.5 * G * (hE + hW) * (zM - zP)
        return f0, f1 + corr_l, f1 + corr_r, f2, f3, f4, f5, source

    def step(self, dt: float) -> None:
        h, hu, hv, hc, z = self.h, self.hu, self.hv, self.hc, self.z
        u, v, c, _ = self.velocities()
        inv = torch.where(h > self.h_dry, 1.0 / h.clamp_min(self.h_dry), torch.zeros_like(h))
        self._ice = (self.hi * inv).clamp(0.0, 1.0)
        self._heat = self.hq * inv
        dx = self.dx
        # x-direction (axis=1): faces between columns
        f0, f1l, f1r, f2, f3, f4, f5, src = self._direction_flux(h, u, v, c, z, 1)
        dh = torch.zeros_like(h); dhu = src.clone(); dhv = torch.zeros_like(h); dhc = torch.zeros_like(h); dhi = torch.zeros_like(h); dhq = torch.zeros_like(h)
        dhi[:, :-1] -= f4; dhi[:, 1:] += f4
        dhq[:, :-1] -= f5; dhq[:, 1:] += f5
        dh[:, :-1] -= f0; dh[:, 1:] += f0
        dhu[:, :-1] -= f1l; dhu[:, 1:] += f1r
        dhv[:, :-1] -= f2; dhv[:, 1:] += f2
        dhc[:, :-1] -= f3; dhc[:, 1:] += f3
        # y-direction (axis=0): rows increase southward; normal velocity is -v
        f0, f1l, f1r, f2, f3, f4, f5, src = self._direction_flux(h, u, v, c, z, 0)
        dhi[:-1, :] -= f4; dhi[1:, :] += f4
        dhq[:-1, :] -= f5; dhq[1:, :] += f5
        dh[:-1, :] -= f0; dh[1:, :] += f0
        dhv[:-1, :] += f1l; dhv[1:, :] -= f1r   # momentum in -v direction flips sign
        dhv -= src
        dhu[:-1, :] -= f2; dhu[1:, :] += f2
        dhc[:-1, :] -= f3; dhc[1:, :] += f3
        k = dt / dx
        h_new = (h + k * dh).clamp_min(0.0)
        hu_new = hu + k * dhu
        hv_new = hv + k * dhv
        hc_new = (hc + k * dhc).clamp_min(0.0)
        hi_new = (self.hi + k * dhi).clamp_min(0.0)
        hq_new = (self.hq + k * dhq).clamp_min(0.0)
        # open boundaries: outflow through edges is allowed by letting edge cells drain
        for arr in (h_new, hu_new, hv_new, hc_new, hi_new, hq_new):
            arr[0, :] = 0.0; arr[-1, :] = 0.0; arr[:, 0] = 0.0; arr[:, -1] = 0.0
        # friction (semi-implicit split)
        wet = h_new > self.h_dry
        hs = h_new.clamp_min(self.h_dry)
        un = hu_new / hs; vn = hv_new / hs
        cn = (hc_new / hs).clamp(0.0, 1.0)
        speed = torch.sqrt(un * un + vn * vn)
        mu = self.mu_s * cn
        n = self.n_w + (self.n_d - self.n_w) * cn
        turb = 1.0 + dt * G * n * n * speed / hs.pow(4.0 / 3.0)
        coul = (1.0 - dt * G * mu / speed.clamp_min(1e-6)).clamp_min(0.0)
        fac = torch.where(wet, coul / turb, torch.zeros_like(h_new))
        hc_new = torch.minimum(hc_new, h_new)
        hi_new = torch.minimum(hi_new, hc_new)
        hu_new = hu_new * fac; hv_new = hv_new * fac
        thin = h_new < 0.02  # films thinner than 2 cm carry no momentum (robustness, CFL)
        hu_new = torch.where(thin, torch.zeros_like(hu_new), hu_new); hv_new = torch.where(thin, torch.zeros_like(hv_new), hv_new)
        if self.melt_eff > 0 or self.melt_tau > 0:
            sp2 = torch.sqrt(hu_new * hu_new + hv_new * hv_new) / hs
            accel = G * mu + G * n * n * sp2 * sp2 / hs.pow(4.0 / 3.0)
            rho = 1000.0 * (1.0 + 0.8 * cn)
            power = rho * h_new * accel * sp2  # W/m2 dissipated by bed friction
            if self.melt_eff > 0:
                hq_new = hq_new + self.melt_eff * power * dt * wet
            LAT = 900.0 * 3.34e5  # J per m3 of ice
            dm = torch.minimum(hi_new, hq_new / LAT)
            if self.melt_tau > 0:
                dm = torch.maximum(dm, hi_new * (1.0 - math.exp(-dt / self.melt_tau)))
            dm = torch.minimum(dm, hi_new) * wet
            hq_new = (hq_new - dm * LAT).clamp_min(0.0)
            hi_new = hi_new - dm
            hc_new = (hc_new - dm).clamp_min(0.0)
            self.melted += float(dm.sum()) * self.dx * self.dx
        if self.erosion_k > 0:
            # bed entrainment: rate K (|u|-u_c)+ limited by the remaining erodible depth
            sp = torch.sqrt(hu_new * hu_new + hv_new * hv_new) / hs
            rate = self.erosion_k * (sp - self.erosion_uc).clamp_min(0.0) * wet * (cn > 0.05)
            de = torch.minimum(rate * dt, self.erodible)
            self.erodible = self.erodible - de
            self.z = self.z - de
            h_new = h_new + de
            hc_new = hc_new + de * self.erosion_c
            self.entrained += float(de.sum()) * self.dx * self.dx
        self.h, self.hu, self.hv, self.hc, self.hi, self.hq = h_new, hu_new, hv_new, hc_new, hi_new, hq_new
        self.time += dt

    def add_volume(self, rows, cols, volumes_m3: torch.Tensor, c_value: float = 0.0) -> None:
        dh = volumes_m3 / (self.dx * self.dx)
        self.h[rows, cols] += dh
        self.hc[rows, cols] += dh * c_value
        self.hq[rows, cols] += dh * 4.2e6 * self.river_temp_c


def encode_frame(model: MixtureSWE, h_min: float = 0.05):
    h = model.h
    u, v, c, _ = model.velocities()
    mask = h > h_min
    idx = torch.nonzero(mask.ravel(), as_tuple=False).ravel()
    speed = torch.sqrt(u * u + v * v)
    return {
        "idx": idx.cpu().numpy().astype(np.int32),
        "h": h.ravel()[idx].cpu().numpy().astype(np.float16),
        "c": c.ravel()[idx].cpu().numpy().astype(np.float16),
        "ice": (model.hi / h.clamp_min(1e-3)).ravel()[idx].cpu().numpy().astype(np.float16),
        "speed": speed.ravel()[idx].cpu().numpy().astype(np.float16),
        "hu": model.hu.ravel()[idx].cpu().numpy().astype(np.float32),
        "hv": model.hv.ravel()[idx].cpu().numpy().astype(np.float32),
    }


def front_chainage(model: MixtureSWE, chain: torch.Tensor, h_thr: float, c_thr: float | None,
                   h_base: torch.Tensor | None = None, rise: float = 0.0) -> float:
    if c_thr is not None:
        _, _, c, _ = model.velocities()
        mask = (model.h > h_thr) & (c > c_thr)
    else:
        mask = (model.h - h_base) > rise
    vals = chain[mask]
    vals = vals[~torch.isnan(vals)]
    return float(vals.max()) if vals.numel() else float("nan")


def run(args: argparse.Namespace) -> None:
    inp = np.load(args.inputs)
    meta = json.loads(Path(args.inputs).with_suffix(".json").read_text("utf-8"))
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    dev = args.device
    model = MixtureSWE(inp["z"], meta["cell_m"], device=dev, mu_s=args.mu_s, n_w=args.n_w,
                       n_d=args.n_d, cfl=args.cfl)
    lateral = torch.as_tensor(inp["lateral_q_m3_s"], dtype=torch.float32, device=dev)
    lat_idx = torch.nonzero(lateral.ravel() > 0).ravel()
    lat_rows, lat_cols = lat_idx // model.z.shape[1], lat_idx % model.z.shape[1]
    lat_q = lateral.ravel()[lat_idx]
    ext_rows = torch.tensor([i["row"] for i in meta["external_inflows"]], device=dev)
    ext_cols = torch.tensor([i["col"] for i in meta["external_inflows"]], device=dev)
    ext_q = torch.tensor([i["q_m3_s"] * args.inflow_scale for i in meta["external_inflows"]],
                         dtype=torch.float32, device=dev)
    lat_q = lat_q * args.inflow_scale
    route_chain = torch.as_tensor(inp["route_chainage_m"], device=dev)
    upper_chain = torch.as_tensor(inp["upper_chainage_m"], device=dev)
    stations = meta["stations"]
    st_rows = torch.tensor([s["row"] for s in stations.values()], device=dev)
    st_cols = torch.tensor([s["col"] for s in stations.values()], device=dev)
    transect_path = Path(args.inputs).with_name(Path(args.inputs).stem + "_transects.json")
    transects = json.loads(transect_path.read_text("utf-8")) if transect_path.exists() else {}
    for tr in transects.values():
        tr["rows_t"] = torch.tensor(tr["rows"], device=dev)
        tr["cols_t"] = torch.tensor(tr["cols"], device=dev)

    model.melt_eff, model.melt_tau = args.melt_eff, args.melt_tau
    model.river_temp_c = args.river_temp_c
    if args.erosion_k > 0:
        model.erosion_k, model.erosion_uc, model.erosion_c = args.erosion_k, args.erosion_uc, args.erosion_c
        channel = torch.as_tensor(inp["channel"], device=dev)
        if args.erodible_zone == "valley":
            # valley floor: within N cells of the channel and less than 30 m above the local channel bed
            k = max(1, int(round(args.erodible_halfwidth_m / model.dx)))
            ch = channel.float()[None, None]
            near = torch.nn.functional.max_pool2d(ch, 2 * k + 1, stride=1, padding=k)[0, 0] > 0
            zch = torch.where(channel, model.z, torch.full_like(model.z, float("inf")))
            zmin = -torch.nn.functional.max_pool2d(-zch[None, None], 2 * k + 1, stride=1, padding=k)[0, 0]
            zone = near & (model.z - zmin < 30.0)
        else:
            zone = channel
        model.erodible = torch.where(zone, torch.full_like(model.z, args.erodible_depth), torch.zeros_like(model.z))
        print(f"entrainment on: K={args.erosion_k} u_c={args.erosion_uc} depth={args.erodible_depth} m, zone={args.erodible_zone} "
              f"({int(zone.sum())} cells, {float(zone.sum()) * model.dx**2 * args.erodible_depth / 1e6:.1f} Mm3 available)")
    if args.restart:
        st = np.load(args.restart)
        for k in ("h", "hu", "hv", "hc"):
            setattr(model, k, torch.as_tensor(st[k], device=dev))
        if "hi" in st:
            model.hi = torch.as_tensor(st["hi"], device=dev)
        model.hq = (model.h - model.hc).clamp_min(0.0) * 4.2e6 * args.river_temp_c
        print(f"restarted from {args.restart}")
    elif args.init_normal_depth:
        # Manning normal depth from in-domain runoff only; boundary inflows fill during spin-up.
        acc = torch.as_tensor(inp["acc_cells"], dtype=torch.float32, device=dev)
        channel = torch.as_tensor(inp["channel"], device=dev)
        q = meta["uniform_runoff_m_s"] * acc * model.dx ** 2 * args.inflow_scale
        gz = torch.gradient(model.z, spacing=model.dx)
        slope = torch.sqrt(gz[0] ** 2 + gz[1] ** 2).clamp_min(1e-4)
        hn = (args.n_w * q / (model.dx * torch.sqrt(slope))) ** 0.6
        model.h = torch.where(channel, hn, torch.zeros_like(hn))
        print(f"normal-depth init: channel volume {float(model.h.sum()) * model.dx**2 / 1e6:.2f} Mm3, max h {float(hn[channel].max()):.2f} m")
    model.time = 0.0
    h_base = model.h.clone()

    release_mask = None
    if args.release:
        release_mask = torch.as_tensor(inp["source_mask"], device=dev)
        thick = float(inp["source_thickness_m"]) * args.release_volume_scale
        if args.release_duration <= 0:
            model.h[release_mask] += thick
            model.hc[release_mask] += thick * args.release_c0
            model.hi[release_mask] += thick * args.release_ice_fraction
        release_rate = thick / max(args.release_duration, 1e-9)  # m/s of thickness added while releasing
        print(f"release {float(release_mask.sum()) * model.dx**2 * thick / 1e6:.2f} Mm3 over {int(release_mask.sum())} cells, "
              f"debris fraction {args.release_c0}, duration {args.release_duration} s")

    series_path = out / "series.csv"
    fields = ["time_s", "dt_s", "wall_s", "total_volume_m3", "debris_volume_m3", "entrained_m3", "ice_volume_m3", "melted_m3", "max_h_m", "max_speed_m_s",
              "debris_front_route_km", "debris_front_upper_km", "flood_front_route_km"]
    for key in stations:
        fields += [f"{key}_h", f"{key}_speed", f"{key}_c", f"{key}_q"]
    for key in transects:
        fields += [f"{key}_Q", f"{key}_Qdebris", f"{key}_hmax", f"{key}_stage", f"{key}_cmax", f"{key}_wet_width_m"]
    writer_file = series_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(writer_file, fieldnames=fields)
    writer.writeheader()

    t_end, frame_dt = args.t_end, args.frame_dt
    next_frame, frame_no = 0.0, 0
    wall0 = time.time()
    step_no = 0
    injected = 0.0
    while model.time < t_end - 1e-9:
        dt = min(model.max_dt(), t_end - model.time, next_frame + 1e-9 - model.time if next_frame > model.time else frame_dt)
        dt = max(dt, 1e-3)
        model.add_volume(lat_rows, lat_cols, lat_q * dt)
        model.add_volume(ext_rows, ext_cols, ext_q * dt)
        injected += float((lat_q.sum() + ext_q.sum()) * dt)
        if release_mask is not None and args.release_duration > 0 and model.time < args.release_duration:
            d_th = release_rate * min(dt, args.release_duration - model.time)
            model.h[release_mask] += d_th
            model.hc[release_mask] += d_th * args.release_c0
            model.hi[release_mask] += d_th * args.release_ice_fraction
        model.step(dt)
        step_no += 1
        if model.time >= next_frame - 1e-6:
            u, v, c, _ = model.velocities()
            speed = torch.sqrt(u * u + v * v)
            row = {
                "time_s": round(model.time, 3), "dt_s": dt, "wall_s": round(time.time() - wall0, 1),
                "total_volume_m3": float(model.h.sum()) * model.dx**2,
                "debris_volume_m3": float(model.hc.sum()) * model.dx**2,
                "entrained_m3": model.entrained,
                "ice_volume_m3": float(model.hi.sum()) * model.dx**2, "melted_m3": model.melted,
                "max_h_m": float(model.h.max()), "max_speed_m_s": float(speed.max()),
                "debris_front_route_km": front_chainage(model, route_chain, 0.1, 0.05) / 1e3,
                "debris_front_upper_km": front_chainage(model, upper_chain, 0.1, 0.05) / 1e3,
                "flood_front_route_km": front_chainage(model, route_chain, 0.0, None, h_base, 0.5) / 1e3,
            }
            for key, r, cc in zip(stations, st_rows.tolist(), st_cols.tolist()):
                hh = model.h[r - 1:r + 2, cc - 1:cc + 2]
                k = int(torch.argmax(hh)); rr, ccc = r - 1 + k // 3, cc - 1 + k % 3
                row[f"{key}_h"] = float(model.h[rr, ccc]); row[f"{key}_speed"] = float(speed[rr, ccc])
                row[f"{key}_c"] = float(c[rr, ccc]); row[f"{key}_q"] = float(model.h[rr, ccc] * speed[rr, ccc] * model.dx)
            for key, tr in transects.items():
                tr_r, tr_c = tr["rows_t"], tr["cols_t"]
                tx, ty = tr["tangent"]
                flux = (model.hu[tr_r, tr_c] * tx + model.hv[tr_r, tr_c] * ty) * model.dx * tr.get("flux_scale", 1.0)
                ht = model.h[tr_r, tr_c]
                ct = c[tr_r, tr_c]
                wet = ht > 0.05
                row[f"{key}_Q"] = float(flux.sum())
                row[f"{key}_Qdebris"] = float((flux * ct).sum())
                row[f"{key}_hmax"] = float(ht.max())
                row[f"{key}_stage"] = float((model.z[tr_r, tr_c] + ht)[wet].min()) if bool(wet.any()) else float("nan")
                row[f"{key}_cmax"] = float(ct[wet].max()) if bool(wet.any()) else 0.0
                row[f"{key}_wet_width_m"] = float(wet.sum()) * model.dx
            writer.writerow(row); writer_file.flush()
            if args.save_frames:
                np.savez_compressed(out / "frames" / f"frame_{frame_no:05d}.npz", time_s=model.time, **encode_frame(model))
            print(f"t={model.time:8.1f}s step={step_no} dt={dt:.3f} wall={row['wall_s']:.0f}s "
                  f"V={row['total_volume_m3']/1e6:.2f}Mm3 hmax={row['max_h_m']:.1f} umax={row['max_speed_m_s']:.1f} "
                  f"debris_front={row['debris_front_route_km']:.1f}km flood_front={row['flood_front_route_km']:.1f}km", flush=True)
            frame_no += 1
            next_frame += frame_dt
    writer_file.close()
    np.savez_compressed(out / "final_state.npz", h=model.h.cpu().numpy(), hu=model.hu.cpu().numpy(),
                        hv=model.hv.cpu().numpy(), hc=model.hc.cpu().numpy(), hi=model.hi.cpu().numpy(), hq=model.hq.cpu().numpy(), time_s=model.time,
                        bed_change=(model.z - model.z0).cpu().numpy())
    result = {"status": "ok", "steps": step_no, "wall_s": time.time() - wall0, "t_end": model.time,
              "injected_m3": injected, "entrained_m3": model.entrained, "melted_m3": model.melted,
              "final_volume_m3": float(model.h.sum()) * model.dx**2,
              "final_debris_m3": float(model.hc.sum()) * model.dx**2, "args": vars(args), "inputs_meta": meta}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("done", json.dumps({k: result[k] for k in ("steps", "wall_s", "final_volume_m3")}))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--restart")
    p.add_argument("--release", action="store_true")
    p.add_argument("--init-normal-depth", action="store_true")
    p.add_argument("--release-c0", type=float, default=1.0, help="initial debris fraction of the release")
    p.add_argument("--release-volume-scale", type=float, default=1.0)
    p.add_argument("--release-ice-fraction", type=float, default=0.0, help="fraction of the release volume that is ice (part of c0)")
    p.add_argument("--melt-eff", type=float, default=0.0, help="fraction of bed-friction dissipation that melts ice")
    p.add_argument("--river-temp-c", type=float, default=12.0, help="temperature of pre-event and inflowing river water")
    p.add_argument("--melt-tau", type=float, default=0.0, help="prescribed ice melt time constant (s), 0 = off")
    p.add_argument("--release-duration", type=float, default=0.0, help="seconds over which the source is released (0 = instant)")
    p.add_argument("--erosion-k", type=float, default=0.0, help="entrainment coefficient (m of bed per m of excess velocity travel)")
    p.add_argument("--erosion-uc", type=float, default=3.0, help="threshold speed for entrainment (m/s)")
    p.add_argument("--erosion-c", type=float, default=0.7, help="debris fraction of entrained bed material")
    p.add_argument("--erodible-depth", type=float, default=8.0)
    p.add_argument("--erodible-zone", choices=["channel", "valley"], default="channel")
    p.add_argument("--erodible-halfwidth-m", type=float, default=150.0)
    p.add_argument("--t-end", type=float, required=True)
    p.add_argument("--frame-dt", type=float, default=60.0)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--mu-s", type=float, default=0.08)
    p.add_argument("--n-w", type=float, default=0.05)
    p.add_argument("--n-d", type=float, default=0.10)
    p.add_argument("--cfl", type=float, default=0.45)
    p.add_argument("--inflow-scale", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
