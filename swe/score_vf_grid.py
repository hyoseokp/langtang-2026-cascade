"""(release volume x ice fraction) inversion: seismic force (30 m short runs with the free-fall impact speed), gorge heights,
flood gauges (routed corridor runs) and the energy bound on melting, scored together. Stored runs only.
Run names: force run fvf30_s{scale}_i{ice} (5 s frames, 900 s); corridor run vf60_s{scale}_i{ice} (ice 0.2: vol60_s{scale}, scale 1: prodfinal60)."""

from __future__ import annotations

import os

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "swe"))
import route1d, sweep  # noqa: E402

SCALES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]; ICES = [0.1, 0.2, 0.3, 0.5]; V0 = 35.4
UPPER = ("gyirong_cctv_arrival", "rasuwagadhi_signal_loss", "syabrubesi_signal_loss")
SEISMIC = [("force_peak_N", 1.55e12, 0.45e12), ("force_duration_s", 88.0, 25.0)]
HEIGHT = [("upper_gorge_hmax_m", 59.0, 11.0), ("rasuwagadhi_rise_m", 27.0, 8.0), ("syabrubesi_rise_m", 27.0, 8.0)]
MELT = [("melt2h_pct", 100.0, 5.0)]  # complete melting: >= 95 % within 2 h scores 1


def corridor_run(scale, ice):
    return f"vf60_s{scale:g}_i{ice:g}"


def force_metrics(run):
    p = ROOT / "figures" / "paper" / f"table_force_history_{run}.csv"
    if not p.exists():
        return np.nan, np.nan
    d = pd.read_csv(p); big = d[d.Fmag > 0.3 * d.Fmag.max()]
    return float(d.Fmag.max()), float(big.t.max() - big.t.min())


def thalweg_path(inp):
    """D8 path from the source with its chainage (km) and 3x3 neighbourhoods, as in render_profile / height_vs_trimline."""
    z = inp["z"]; rows, cols = z.shape; cell = 30.0 if rows > 1500 or cols < 1000 else 60.0
    rcv = inp["rcv"].astype(np.int64).ravel(); src = inp["source_mask"]
    r0, c0 = np.argwhere(src)[np.argmin(z[src])]; node = int(r0 * cols + c0); path = [node]
    while rcv[node] >= 0:
        node = int(rcv[node]); path.append(node)
    path = np.array(path); pr, pc = np.divmod(path, cols)
    step = np.where((np.diff(pr) != 0) & (np.diff(pc) != 0), cell * np.sqrt(2.0), cell); chain = np.r_[0.0, np.cumsum(step)] / 1e3
    nb = np.stack([np.clip(pr + dr, 0, rows - 1) * cols + np.clip(pc + dc, 0, cols - 1) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=1)
    return chain, nb


def upper_gorge_hmax(run, inp, limit_km=22.0):
    """Median over 1-km bins of the maximum flow height sampled on the thalweg path (3x3), the metric of the trimline comparison."""
    chain, nb = thalweg_path(inp); n = inp["z"].size; hmax = np.zeros(len(chain), np.float32)
    for fp in sorted((ROOT / "runs" / run / "frames").glob("frame_*.npz")):
        fr = np.load(fp); h = np.zeros(n, np.float32); h[fr["idx"]] = fr["h"].astype(np.float32); hmax = np.maximum(hmax, h[nb].max(axis=1))
    sel = chain < limit_km
    if not sel.any():
        return np.nan
    b = np.floor(chain[sel]).astype(int); peaks = np.zeros(b.max() + 1, np.float32); np.maximum.at(peaks, b, hmax[sel])
    return float(np.median(peaks[peaks > 0]))


def main() -> None:
    inp30 = dict(np.load(ROOT / "inputs" / "upper30h.npz"))  # gorge heights from the 30 m force runs (resolution)
    rows = []
    for ice in ICES:
        for sc in SCALES:
            rec = {"scale": sc, "volume_Mm3": sc * V0, "ice": ice}; terms = {}
            fpk, fdur = force_metrics(f"fvf30_s{sc:g}_i{ice:g}")
            for k, tgt, tol in SEISMIC:
                v = fpk if k == "force_peak_N" else fdur; terms[k] = 9.0 if not np.isfinite(v) else min(((v - tgt) / tol) ** 2, 9.0); rec[k] = v
            run = corridor_run(sc, ice); p = ROOT / "runs" / run / "series.csv"
            if p.exists() and (ROOT / "runs" / run / "result.json").exists():
                out = f"r1d_{run}_b3_f2_s8"
                if not (ROOT / "runs" / out / "series.csv").exists():
                    a = SimpleNamespace(tag="corridor60s", start="syabrubesi_signal_loss", driver=run, out=out, n_w=0.03, n_d=0.02, dep_uc=0.0, dep_tau=600.0,
                                        width_scale=1.0, h_bank=3.0, fp_scale=2.0, t_end=28800.0, frame_dt=60.0, cfl=0.5, spin=28800.0, quiet=True)
                    with contextlib.redirect_stdout(io.StringIO()):
                        route1d.run(a)
                s2 = pd.read_csv(p); s1 = pd.read_csv(ROOT / "runs" / out / "series.csv")
                for st, qty, target, tol in sweep.CRITERIA:
                    v = sweep.station_metrics(s2 if st in UPPER else s1, st)[qty]; terms[f"{st.split('_')[0]}_{qty}"] = 9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0)
                    rec[f"{st.split('_')[0]}_{qty}"] = round(v, 2) if np.isfinite(v) else np.nan
                f30 = f"fvf30_s{sc:g}_i{ice:g}"; s30 = pd.read_csv(ROOT / "runs" / f30 / "series.csv") if (ROOT / "runs" / f30 / "series.csv").exists() else None
                hts = {"upper_gorge_hmax_m": upper_gorge_hmax(f30, inp30) if (ROOT / "runs" / f30 / "frames").exists() else np.nan,
                       "rasuwagadhi_rise_m": sweep.station_metrics(s30, "rasuwagadhi_signal_loss")["rise_m"] if s30 is not None else np.nan,
                       "syabrubesi_rise_m": sweep.station_metrics(s2, "syabrubesi_signal_loss")["rise_m"]}
                for k, tgt, tol in HEIGHT:
                    v = hts[k]; terms[k] = 9.0 if not np.isfinite(v) else min(((v - tgt) / tol) ** 2, 9.0); rec[k] = v
                t = s2.time_s.to_numpy(); m = s2.melted_m3.to_numpy(); m2h = 100 * m[int(np.argmin(np.abs(t - 7200)))] / (sc * V0 * 1e6 * ice)
                terms["melt2h_pct"] = min(((max(m2h, 0) - 100.0) / 5.0) ** 2, 9.0) if m2h < 95 else 0.0; rec["melt2h_pct"] = m2h
            groups = {"seismic": [k for k, _, _ in SEISMIC], "heights": [k for k, _, _ in HEIGHT], "melt": ["melt2h_pct"],
                      "gauges": [k for k in terms if k.split("_")[0] in ("gyirong", "rasuwagadhi", "syabrubesi", "betrawati", "galchhi", "malekhu", "kali", "devghat")]}
            for g, keys in groups.items():
                vals = [terms[k] for k in keys if k in terms]; rec[f"misfit_{g}"] = np.mean(vals) if vals else np.nan
            rec["misfit"] = np.nanmean([rec[f"misfit_{g}"] for g in groups]); rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(ROOT / "figures" / "paper" / "table_vf_grid.csv", index=False)
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.8), dpi=150)
    for ax, key, title in zip(axes, ("misfit", "misfit_seismic", "misfit_heights", "misfit_gauges", "misfit_melt"), ("all groups", "seismic force", "gorge heights", "flood gauges", "complete melting")):
        Z = df.pivot(index="ice", columns="volume_Mm3", values=key).values
        im = ax.imshow(Z, origin="lower", cmap="viridis_r", vmin=0, vmax=min(np.nanmax(Z) if np.isfinite(Z).any() else 1, 9))
        ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([f"{s*V0:.0f}" for s in SCALES]); ax.set_yticks(range(len(ICES))); ax.set_yticklabels([f"{i:g}" for i in ICES])
        ax.set_xlabel("release volume (10$^6$ m$^3$)"); ax.set_ylabel("ice fraction"); ax.set_title(title, fontsize=9)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w" if Z[i, j] > 0.6 * np.nanmax(Z) else "k")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Inversion for release volume and ice fraction (misfit in tolerance units)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_vf_grid.png"); plt.close(fig)
    pd.set_option("display.width", 260); print(df[["volume_Mm3", "ice", "misfit", "misfit_seismic", "misfit_heights", "misfit_gauges", "misfit_melt", "force_peak_N", "force_duration_s", "upper_gorge_hmax_m", "devghat_excess_Mm3", "melt2h_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
