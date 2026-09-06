"""1-D Saint-Venant routing below Syabrubesi on the synthetic-channel profile, driven by the 2-D model's Syabrubesi hydrograph.

Variable-width rectangular channel (width from contributing area as in synth_channel), HLL, hydrostatic
reconstruction, Manning n(c), solids tracer with settling. Lateral baseflow and the Budhi Gandaki / Marsyangdi
inflows enter at their junction chainages. Writes runs/<out>/series.csv with the same station columns as the 2-D runs.
"""

from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "swe"))
from synth_channel import width_from_area  # noqa: E402

G = 9.81
STATIONS = ["betrawati_gauge", "galchhi_gauge", "malekhu_gauge", "kali_khola_gauge", "devghat_2023_candidate"]


def build(tag: str, start: str, width_scale: float):
    d = np.load(ROOT / "inputs" / f"{tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    z = d["z"].astype(np.float64); rows, cols = z.shape; cell = meta["cell_m"]
    rcv = d["rcv"].astype(np.int64).ravel(); acc = d["acc_cells"].astype(np.float64).ravel() * cell * cell / 1e6
    st = meta["stations"][start]; node = st["row"] * cols + st["col"]; path = [node]
    while rcv[node] >= 0:
        node = int(rcv[node]); path.append(node)
    path = np.array(path); pr, pc = np.divmod(path, cols)
    step = np.where((np.diff(pr) != 0) & (np.diff(pc) != 0), cell * np.sqrt(2), cell); chain = np.r_[0.0, np.cumsum(step)]
    zp = z.ravel()[path]
    zp = np.minimum.accumulate(zp)  # monotone (the synthetic profile already is; guards the DSM part)
    W = width_from_area(acc[path]) * width_scale
    idx_of = np.full(rows * cols, -1, np.int64); idx_of[path] = np.arange(len(path))

    def junction(r, c):  # follow D8 from a cell until it meets the path
        n = r * cols + c
        for _ in range(rows * cols):
            if idx_of[n] >= 0:
                return int(idx_of[n])
            if rcv[n] < 0:
                return -1
            n = int(rcv[n])
        return -1

    tr = json.loads((ROOT / "inputs" / f"{tag}_transects.json").read_text("utf-8"))
    stations = {}
    for key in STATIONS:
        if key in meta["stations"]:
            r, c = meta["stations"][key]["row"], meta["stations"][key]["col"]
        else:  # extra gauges live only in the transect file: use the transect cell with the largest contributing area
            rr, cc = np.array(tr[key]["rows"]), np.array(tr[key]["cols"]); k = int(np.argmax(acc[rr * cols + cc])); r, c = int(rr[k]), int(cc[k])
        j = junction(r, c)
        if j >= 0:
            stations[key] = j
    ext = []
    for inf in meta["external_inflows"]:
        j = junction(inf["row"], inf["col"])
        if j > 0:
            ext.append((j, inf["q_m3_s"], inf.get("name", "")))
    # lateral baseflow from the growth of contributing area, excluding the external-inflow jumps
    runoff = meta["uniform_runoff_m_s"] * 1e6  # m3/s per km2
    dacc = np.r_[0.0, np.diff(acc[path])].clip(min=0)
    for j, q, _ in ext:
        dacc[j] = 0.0  # the tributary's own area is represented by the measured inflow
    q_lat = runoff * dacc  # m3/s per cell
    # resample onto the true route chainage: the 4-connected D8 path is ~1.4x longer than the mapped channel
    s_cells = d["route_chainage_m"].astype(np.float64).ravel()[path]
    ok = np.isfinite(s_cells)
    s_rel = np.interp(np.arange(len(path)), np.nonzero(ok)[0], s_cells[ok]) - np.nanmin(s_cells[ok][:5])
    s_rel = np.maximum.accumulate(np.maximum(s_rel, 0.0))
    s_u = np.arange(0.0, s_rel.max(), cell)
    z_u = np.minimum.accumulate(np.interp(s_u, s_rel, zp)); W_u = np.interp(s_u, s_rel, W)
    q_u = np.histogram(s_rel, bins=np.r_[s_u, s_u[-1] + cell], weights=q_lat)[0]
    to_u = lambda j: int(min(np.searchsorted(s_u, s_rel[j]), len(s_u) - 1))
    stations = {k: to_u(j) for k, j in stations.items()}
    ext = [(to_u(j), q, nm) for j, q, nm in ext]
    if False: print(f"path {len(path)} cells = {chain[-1]/1e3:.1f} km on the grid, {s_u[-1]/1e3:.1f} km along the route (ratio {chain[-1]/s_u[-1]:.2f})")
    return s_u, z_u, W_u, stations, ext, q_u, np.full(len(s_u), cell)


def run(a):
    chain, zb, W, stations, ext, q_lat, step = build(a.tag, a.start, a.width_scale)
    n = len(zb); dx = step
    # boundary hydrograph from the 2-D run at the start station
    s2 = pd.read_csv(ROOT / "runs" / a.driver / "series.csv")
    t2 = s2.time_s.to_numpy(); Qin = np.abs(s2[f"{a.start}_Q"].to_numpy()); Qd = np.abs(s2[f"{a.start}_Qdebris"].to_numpy())
    cin = np.where(Qin > 1, Qd / np.maximum(Qin, 1), 0.0)
    q_ext = np.zeros(n)
    for j, q, _ in ext:
        q_ext[j] += q
    Qbase = Qin[0] + np.cumsum(q_lat + q_ext)
    slope = np.maximum(-np.gradient(zb, chain), 1e-4)
    h = (a.n_w * Qbase / (W * np.sqrt(slope))) ** 0.6
    u = Qbase / (W * h); hc = np.zeros(n); dz = np.zeros(n)
    Wf = 0.5 * (W[:-1] + W[1:])
    Wfp = a.fp_scale * W; hb = h + a.h_bank  # bank height measured above the pre-event water depth

    def area(hh):  # wetted area per unit length: main channel plus floodplain storage above the bank
        return W * hh + Wfp * np.maximum(hh - hb, 0.0)

    def depth(A):
        return np.where(A <= W * hb, A / W, hb + (A - W * hb) / (W + Wfp))

    out_rows = []; t = -a.spin; next_out = 0.0; deposited = 0.0
    while t < a.t_end:
        c = np.clip(hc / np.maximum(h, 1e-3), 0, 1)
        smax = np.max(np.abs(u) + np.sqrt(G * h)); dt = min(a.cfl * dx.min() / max(smax, 1e-6), 5.0, a.t_end - t)
        # boundary: inflow at cell 0 imposed by mass source
        qin = np.interp(max(t, 0.0), t2, Qin); ci = np.interp(max(t, 0.0), t2, cin)
        # HR at faces
        zf = np.maximum(zb[:-1], zb[1:]); hL = np.maximum(h[:-1] + zb[:-1] - zf, 0); hR = np.maximum(h[1:] + zb[1:] - zf, 0)
        uL, uR = u[:-1], u[1:]
        cL_, cR_ = np.sqrt(G * hL), np.sqrt(G * hR)
        sL = np.minimum(uL - cL_, uR - cR_); sR = np.maximum(uL + cL_, uR + cR_)
        fL0, fR0 = hL * uL, hR * uR; fL1 = hL * uL ** 2 + 0.5 * G * hL ** 2; fR1 = hR * uR ** 2 + 0.5 * G * hR ** 2
        den = np.where(np.abs(sR - sL) > 1e-8, sR - sL, 1.0)
        f0 = np.where(sL >= 0, fL0, np.where(sR <= 0, fR0, (sR * fL0 - sL * fR0 + sL * sR * (hR - hL)) / den))
        f1 = np.where(sL >= 0, fL1, np.where(sR <= 0, fR1, (sR * fL1 - sL * fR1 + sL * sR * (hR * uR - hL * uL)) / den))
        cf = np.where(f0 >= 0, c[:-1], c[1:])
        # cell updates (conservative in W h, W h u)
        dm = np.zeros(n); dmom = np.zeros(n); dc = np.zeros(n)
        dm[:-1] -= Wf * f0; dm[1:] += Wf * f0
        dc[:-1] -= Wf * f0 * cf; dc[1:] += Wf * f0 * cf
        corrL = 0.5 * G * (h[:-1] ** 2 - hL ** 2); corrR = 0.5 * G * (h[1:] ** 2 - hR ** 2)
        dmom[:-1] -= Wf * (f1 + corrL); dmom[1:] += Wf * (f1 + corrR)
        # bank pressure from width change
        Wp = np.r_[Wf, W[-1]]; Wm = np.r_[W[0], Wf]
        dmom += 0.5 * G * h ** 2 * (Wp - Wm)
        dm += q_lat + q_ext; dm[0] += qin; dc[0] += qin * ci
        # outflow at the last cell: free outflow (copy flux)
        dm[-1] -= W[-1] * h[-1] * max(u[-1], 0); dmom[-1] -= W[-1] * (h[-1] * u[-1] ** 2) * (u[-1] > 0)
        dc[-1] -= W[-1] * h[-1] * max(u[-1], 0) * c[-1]
        A_new = np.maximum(area(h) + dt * dm / dx, 0.0); mom = W * h * u + dt * dmom / dx; hcW = W * hc + dt * dc / dx
        h = depth(A_new); hc = np.clip(hcW / W, 0.0, h)
        u = np.where(h > 1e-3, mom / (W * np.maximum(h, 1e-3)), 0.0)
        cn = np.clip(hc / np.maximum(h, 1e-3), 0, 1); nn = a.n_w + (a.n_d - a.n_w) * cn
        u = u / (1.0 + dt * G * nn ** 2 * np.abs(u) / np.maximum(h, 1e-3) ** (4 / 3))
        if a.dep_uc > 0:
            rate = np.clip(1 - np.abs(u) / a.dep_uc, 0, 1) * (cn > 0.05) * (1 - np.exp(-dt / a.dep_tau))
            dd = hc * rate; hc -= dd; h -= dd; dz += dd; zb = zb + dd; deposited += float((dd * W * dx).sum())
        t += dt
        if t >= next_out - 1e-6 and t >= -1e-6:
            row = {"time_s": round(t, 1), "total_volume_m3": float((area(h) * dx).sum()), "debris_volume_m3": float((hc * W * dx).sum()), "deposited_m3": deposited}
            for key, j in stations.items():
                row[f"{key}_Q"] = float(W[j] * h[j] * u[j]); row[f"{key}_Qdebris"] = float(W[j] * hc[j] * u[j])
                row[f"{key}_stage"] = float(zb[j] + h[j]); row[f"{key}_hmax"] = float(h[j]); row[f"{key}_cmax"] = float(cn[j]); row[f"{key}_wet_width_m"] = float(W[j])
            out_rows.append(row); next_out += a.frame_dt
    out = ROOT / "runs" / a.out; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows); df.to_csv(out / "series.csv", index=False)
    json.dump({"args": vars(a), "entrained_m3": 0.0, "deposited_m3": deposited, "stations_km": {k: float(chain[j] / 1e3) for k, j in stations.items()},
               "inflows": [(float(chain[j] / 1e3), q, nm) for j, q, nm in ext]}, (out / "result.json").open("w"), indent=1)
    for key, j in stations.items():
        q = df[f"{key}_Q"].to_numpy(); stg = df[f"{key}_stage"].to_numpy(); tt = df.time_s.to_numpy()
        i = np.nonzero((stg - stg[0] > 0.5) | (q > 2 * q[0] + 50))[0]; ip = int(np.argmax(q))
        print(f"{key:24s} km {chain[j]/1e3:6.1f} W {W[j]:4.0f} arr {tt[i[0]]/3600 if len(i) else np.nan:5.2f} h peak {q[ip]:7.0f} @ {tt[ip]/3600:.2f} h rise {stg.max()-stg[0]:5.1f} m base {q[0]:5.0f}")
    print("deposited Mm3", deposited / 1e6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="corridor60s"); p.add_argument("--start", default="syabrubesi_signal_loss")
    p.add_argument("--driver", default="sw1_00"); p.add_argument("--out", default="r1d_test")
    p.add_argument("--n-w", type=float, default=0.03); p.add_argument("--n-d", type=float, default=0.02)
    p.add_argument("--dep-uc", type=float, default=0.0); p.add_argument("--dep-tau", type=float, default=600.0)
    p.add_argument("--width-scale", type=float, default=1.0); p.add_argument("--t-end", type=float, default=28800.0)
    p.add_argument("--frame-dt", type=float, default=60.0); p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--spin", type=float, default=7200.0, help="pre-event routing time with base inflow (s)")
    p.add_argument("--h-bank", type=float, default=1e9, help="bank height above the pre-event water level at which floodplain storage engages (m)")
    p.add_argument("--fp-scale", type=float, default=0.0, help="floodplain storage width as a multiple of the channel width")
    p.add_argument("--quiet", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
