"""Ice-fraction sweep with the full heat budget: route each run with the fitted storage section, score, tabulate melt.
Writes figures/paper/table_ice_fraction.csv and fig_ice_fraction.png and fig_melt_history.png."""

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
RUNS = {0.1: "icef_m0_i0.1", 0.2: "prodfinal60", 0.3: "icef_m0_i0.3", 0.4: "icef_m0_i0.4", 0.5: "icef_m0_i0.5", 0.65: "icef_m0_i0.65", 0.8: "icef_m0_i0.8"}
V_REL = 35.4e6


def main() -> None:
    rows = []; hist = {}
    for ice, run in RUNS.items():
        p = ROOT / "runs" / run / "series.csv"
        if not p.exists() or not (ROOT / "runs" / run / "result.json").exists():
            continue
        out = f"r1d_{run}_b3_f2_s8"
        if not (ROOT / "runs" / out / "series.csv").exists():
            a = SimpleNamespace(tag="corridor60s", start="syabrubesi_signal_loss", driver=run, out=out, n_w=0.03, n_d=0.02, dep_uc=0.0, dep_tau=600.0,
                                width_scale=1.0, h_bank=3.0, fp_scale=2.0, t_end=28800.0, frame_dt=60.0, cfl=0.5, spin=28800.0, quiet=True)
            with contextlib.redirect_stdout(io.StringIO()):
                route1d.run(a)
        s2 = pd.read_csv(p); s1 = pd.read_csv(ROOT / "runs" / out / "series.csv"); tu, td = [], []; rec = {"ice": ice, "run": run}
        for st, qty, target, tol in sweep.CRITERIA:
            v = sweep.station_metrics(s2 if st in UPPER else s1, st)[qty]; term = 9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0)
            (tu if st in UPPER else td).append(term); rec[f"{st.split('_')[0]}_{qty}"] = round(v, 2) if np.isfinite(v) else np.nan
        res = json.loads((ROOT / "runs" / run / "result.json").read_text("utf-8")); t = s2.time_s.to_numpy(); m = s2.melted_m3.to_numpy()
        rec.update({"misfit_upper": np.mean(tu), "misfit_lower": np.mean(td), "melted_Mm3": res["melted_m3"] / 1e6,
                    "melt2h_pct": 100 * m[int(np.argmin(np.abs(t - 7200)))] / (V_REL * ice), "melt8h_pct": 100 * res["melted_m3"] / (V_REL * ice)})
        rows.append(rec); hist[ice] = (t, m)
    df = pd.DataFrame(rows).sort_values("ice"); df.to_csv(ROOT / "figures" / "paper" / "table_ice_fraction.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), dpi=150)
    ax[0].plot(df.ice, df.misfit_upper, "o-", color="tab:green", label="upper corridor timing (3)"); ax[0].plot(df.ice, df.misfit_lower, "s-", color="tab:red", label="lower gauges (12)")
    ax[0].set_ylabel("misfit"); ax[0].set_ylim(0, 3); ax[0].legend(fontsize=8); ax[0].set_title("fit to the record", fontsize=9)
    ax[1].plot(df.ice, df.melt2h_pct, "o-", color="tab:blue", label="by 2 h"); ax[1].plot(df.ice, df.melt8h_pct, "s--", color="tab:cyan", label="by 8 h")
    ax[1].axhline(95, color="k", ls="--", lw=0.8); ax[1].set_ylabel("release ice melted (%)"); ax[1].set_ylim(0, 105); ax[1].legend(fontsize=8, loc="lower left"); ax[1].set_title("energy bound on melting", fontsize=9)
    ax[2].plot(df.ice, df.devghat_excess_Mm3, "o-", color="tab:purple"); ax[2].axhspan(10, 30, color="gold", alpha=0.3); ax[2].set_ylim(0, 35)
    ax[2].set_ylabel("Devghat excess volume (10$^6$ m$^3$)"); ax[2].set_title("flood volume at 180 km", fontsize=9)
    for a in ax:
        a.set_xlabel("ice fraction of the release (by volume)"); a.set_xlim(0.05, 0.85); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_ice_fraction.png"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ice, (t, m) in sorted(hist.items()):
        ax[0].plot(t / 60, m / 1e6, label=f"ice {ice:g}"); ax[1].plot(t / 60, 100 * m / (V_REL * ice), label=f"ice {ice:g}")
    for a in ax:
        a.set_xlim(0, 120); a.grid(alpha=0.3); a.set_xlabel("time after onset (min)")
        for x in (7, 19, 50):
            a.axvline(x, color="grey", lw=0.6, ls=":")
    ax[0].set_ylabel("melted ice (10$^6$ m$^3$)"); ax[1].set_ylabel("fraction of the release ice melted (%)"); ax[1].set_ylim(0, 100); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_melt_history.png"); plt.close(fig)
    pd.set_option("display.width", 250); print(df[["ice", "run", "misfit_upper", "misfit_lower", "melted_Mm3", "melt2h_pct", "melt8h_pct", "devghat_excess_Mm3"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
