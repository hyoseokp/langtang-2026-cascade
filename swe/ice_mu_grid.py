"""(friction x ice fraction) grid: route each corridor run below Syabrubesi with the fitted storage section, score the
15 criteria (upper from the corridor series, lower from the routed series) and draw the misfit colormap.
Reads stored runs only; routes only runs that have not been routed yet."""

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

MUS = [0, 0.002, 0.005, 0.01, 0.02]; ICES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8]
UPPER = ("gyirong_cctv_arrival", "rasuwagadhi_signal_loss", "syabrubesi_signal_loss")


def run_name(mu, ice):
    return f"imu60e_m{mu:g}_i{ice:g}"


def main() -> None:
    M = np.full((len(ICES), len(MUS)), np.nan); Mu = M.copy(); Md = M.copy(); rows = []
    for j, mu in enumerate(MUS):
        for i, ice in enumerate(ICES):
            run = run_name(mu, ice); p = ROOT / "runs" / run / "series.csv"
            if not p.exists() or not (ROOT / "runs" / run / "result.json").exists():
                continue
            out = f"r1d_{run}_b3_f2_s8"
            if not (ROOT / "runs" / out / "series.csv").exists():
                a = SimpleNamespace(tag="corridor60s", start="syabrubesi_signal_loss", driver=run, out=out, n_w=0.03, n_d=0.02, dep_uc=0.0, dep_tau=600.0,
                                    width_scale=1.0, h_bank=3.0, fp_scale=2.0, t_end=28800.0, frame_dt=60.0, cfl=0.5, spin=28800.0, quiet=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    route1d.run(a)
            s2 = pd.read_csv(p); s1 = pd.read_csv(ROOT / "runs" / out / "series.csv")
            tu, td = [], []; rec = {"mu": mu, "ice": ice}
            for st, qty, target, tol in sweep.CRITERIA:
                src = s2 if st in UPPER else s1
                v = sweep.station_metrics(src, st)[qty]; term = 9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0)
                (tu if st in UPPER else td).append(term); rec[f"{st.split('_')[0]}_{qty}"] = round(v, 2) if np.isfinite(v) else np.nan
            res = json.loads((ROOT / "runs" / run / "result.json").read_text("utf-8"))
            rec.update({"misfit_upper": np.mean(tu), "misfit_lower": np.mean(td), "misfit": np.mean(tu + td),
                        "melted_Mm3": res["melted_m3"] / 1e6, "deposited_Mm3": res.get("deposited_m3", 0) / 1e6})
            M[i, j] = rec["misfit"]; Mu[i, j] = rec["misfit_upper"]; Md[i, j] = rec["misfit_lower"]; rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(ROOT / "figures" / "paper" / "table_ice_mu_grid.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    for ax, (Z, title) in zip(axes, ((M, "all 15 criteria"), (Mu, "upper corridor timing (3)"), (Md, "lower gauges (12)"))):
        vmax = min(np.nanmax(Z), 9) if np.isfinite(Z).any() else 1
        im = ax.imshow(Z, origin="lower", cmap="viridis_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(MUS))); ax.set_xticklabels([f"{m:g}" for m in MUS], rotation=45); ax.set_yticks(range(len(ICES))); ax.set_yticklabels([f"{c:g}" for c in ICES])
        ax.set_xlabel(r"basal friction $\mu_s$"); ax.set_ylabel("ice fraction of the release (by volume)"); ax.set_title(title, fontsize=9)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w" if Z[i, j] > 0.6 * vmax else "k")
        fig.colorbar(im, ax=ax, shrink=0.85, label="misfit")
    fig.suptitle("Misfit against the record over basal friction and ice fraction (release = rock + ice, no initial water)", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_ice_mu_grid.png"); plt.close(fig)
    pd.set_option("display.width", 250)
    print(df[["mu", "ice", "misfit", "misfit_upper", "misfit_lower", "melted_Mm3", "devghat_excess_Mm3", "devghat_rise_m", "galchhi_rise_m", "kali_rise_m"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
