"""Release-volume inversion: route each volume run with the fitted storage section, score the 15 gauge criteria plus three
gorge-height criteria (upper-gorge trimline 48-70 m over 0-22 km, Planet trimline 27 m at Rasuwagadhi and Syabrubesi),
and plot misfit against release volume with the misfit <= min + 1 band as the uncertainty."""

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

UPPER = ("gyirong_cctv_arrival", "rasuwagadhi_signal_loss", "syabrubesi_signal_loss")
RUNS = {0.5: "vol60_s0.5", 0.75: "vol60_s0.75", 1.0: "prodfinal60", 1.5: "vol60_s1.5", 2.0: "vol60_s2", 3.0: "vol60_s3"}
V0 = 35.4
HEIGHT = [("upper_gorge_hmax_m", 59.0, 11.0), ("rasuwagadhi_rise_m", 27.0, 8.0), ("syabrubesi_rise_m", 27.0, 8.0)]


def upper_gorge_hmax(run: str, uc: np.ndarray) -> float:
    hmax = np.zeros(uc.size, np.float32)
    for fp in sorted((ROOT / "runs" / run / "frames").glob("frame_*.npz")):
        fr = np.load(fp); np.maximum.at(hmax, fr["idx"], fr["h"].astype(np.float32))
    sel = np.isfinite(uc) & (uc < 22000) & (hmax > 1)
    return float(np.median(hmax[sel])) if sel.any() else np.nan


def main() -> None:
    uc = np.load(ROOT / "inputs" / "corridor60s.npz")["upper_chainage_m"].astype(np.float64).ravel()
    rows = []
    for sc, run in RUNS.items():
        p = ROOT / "runs" / run / "series.csv"
        if not p.exists() or not (ROOT / "runs" / run / "result.json").exists():
            continue
        out = f"r1d_{run}_b3_f2_s8"
        if not (ROOT / "runs" / out / "series.csv").exists():
            a = SimpleNamespace(tag="corridor60s", start="syabrubesi_signal_loss", driver=run, out=out, n_w=0.03, n_d=0.02, dep_uc=0.0, dep_tau=600.0,
                                width_scale=1.0, h_bank=3.0, fp_scale=2.0, t_end=28800.0, frame_dt=60.0, cfl=0.5, spin=28800.0, quiet=True)
            with contextlib.redirect_stdout(io.StringIO()):
                route1d.run(a)
        s2 = pd.read_csv(p); s1 = pd.read_csv(ROOT / "runs" / out / "series.csv"); terms = {}; rec = {"scale": sc, "volume_Mm3": sc * V0, "run": run}
        for st, qty, target, tol in sweep.CRITERIA:
            v = sweep.station_metrics(s2 if st in UPPER else s1, st)[qty]; terms[f"{st.split('_')[0]}_{qty}"] = 9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0)
            rec[f"{st.split('_')[0]}_{qty}"] = round(v, 2) if np.isfinite(v) else np.nan
        hts = {"upper_gorge_hmax_m": upper_gorge_hmax(run, uc) if (ROOT / "runs" / run / "frames").exists() else np.nan,
               "rasuwagadhi_rise_m": sweep.station_metrics(s2, "rasuwagadhi_signal_loss")["rise_m"], "syabrubesi_rise_m": sweep.station_metrics(s2, "syabrubesi_signal_loss")["rise_m"]}
        for k, target, tol in HEIGHT:
            v = hts[k]; terms[k] = 9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0); rec[k] = round(v, 1) if np.isfinite(v) else np.nan
        rec["misfit_timing_upper"] = np.mean([terms[k] for k in terms if k.split("_")[0] in ("gyirong", "rasuwagadhi", "syabrubesi") and k.endswith("arr_s")])
        rec["misfit_heights"] = np.mean([terms[k] for k, _, _ in HEIGHT]); rec["misfit_lower"] = np.mean([v for k, v in terms.items() if k.split("_")[0] in ("betrawati", "galchhi", "malekhu", "kali", "devghat")])
        rec["misfit"] = np.mean(list(terms.values())); rows.append(rec)
    df = pd.DataFrame(rows).sort_values("scale"); df.to_csv(ROOT / "figures" / "paper" / "table_volume_inversion.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(df.volume_Mm3, df.misfit, "ko-", label="all 18 criteria"); ax.plot(df.volume_Mm3, df.misfit_timing_upper, "s--", color="tab:green", label="upper timing (3)")
    ax.plot(df.volume_Mm3, df.misfit_heights, "^--", color="tab:orange", label="gorge heights (3)"); ax.plot(df.volume_Mm3, df.misfit_lower, "v--", color="tab:red", label="lower gauges (12)")
    best = df.misfit.min(); ok = df[df.misfit <= best + 1]
    ax.axhspan(0, best + 1, color="gold", alpha=0.25); ax.set_xscale("log"); ax.set_xlabel("release volume (10$^6$ m$^3$)"); ax.set_ylabel("misfit"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"Volume inversion: misfit within 1 of the minimum for {ok.volume_Mm3.min():.0f}-{ok.volume_Mm3.max():.0f} Mm3", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_volume_inversion.png"); plt.close(fig)
    pd.set_option("display.width", 250)
    print(df[["volume_Mm3", "misfit", "misfit_timing_upper", "misfit_heights", "misfit_lower", "gyirong_arr_s", "syabrubesi_arr_s", "upper_gorge_hmax_m", "rasuwagadhi_rise_m", "galchhi_rise_m", "devghat_rise_m", "devghat_excess_Mm3"]].round(2).to_string(index=False))
    print(f"volume within misfit min+1: {ok.volume_Mm3.min():.0f}-{ok.volume_Mm3.max():.0f} Mm3 (best {df.loc[df.misfit.idxmin(), 'volume_Mm3']:.0f})")


if __name__ == "__main__":
    main()
