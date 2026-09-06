"""Hold-out test of the routing: fit the two storage parameters on Galchhi and Devghat only, predict Betrawati,
Malekhu and Kali Khola. Also the misfit surface over (bank height, storage width) for identifiability.
Reads the stored 1-D sweep runs only."""

from __future__ import annotations

import os

import glob
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "swe"))
import sweep  # noqa: E402

FIT = [c for c in sweep.CRITERIA if c[0] in ("galchhi_gauge", "devghat_2023_candidate")]
PRED = [c for c in sweep.CRITERIA if c[0] in ("betrawati_gauge", "malekhu_gauge", "kali_khola_gauge")]


def terms(s, crit):
    out = []
    for st, qty, target, tol in crit:
        v = sweep.station_metrics(s, st)[qty]
        out.append((st, qty, v, target, tol, 9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0)))
    return out


def main() -> None:
    driver = sys.argv[1] if len(sys.argv) > 1 else "sw1_00"; prefix = sys.argv[2] if len(sys.argv) > 2 else "fp2"
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "sweep" / f"sw1d_{driver}_{prefix}_scores_*.csv"))], ignore_index=True)
    suf = "_s8" if prefix == "fp2" else ""
    df["run"] = [(f"r1d_{driver}_n{r.n_w:.3f}_w1.0_b{r.h_bank:g}_f{r.fp_scale:g}_d{r.dep_uc:.0f}" if prefix == "fp2" else f"r1d_{driver}_n{r.n_w:.3f}_w1.0_b{r.h_bank:.0f}_f{r.fp_scale:.0f}_d{r.dep_uc:.0f}") + suf for r in df.itertuples()]
    rows = []
    for r in df.itertuples():
        s = pd.read_csv(ROOT / "runs" / r.run / "series.csv")
        tf = terms(s, FIT); tp = terms(s, PRED)
        rows.append({"run": r.run, "n_w": r.n_w, "h_bank": r.h_bank, "fp_scale": r.fp_scale, "dep_uc": r.dep_uc,
                     "misfit_fit": np.mean([t[-1] for t in tf]), "misfit_pred": np.mean([t[-1] for t in tp]),
                     "pass_fit": sum(t[-1] <= 1 for t in tf), "pass_pred": sum(t[-1] <= 1 for t in tp),
                     **{f"{t[0].split('_')[0]}_{t[1]}": round(t[2], 2) for t in tf + tp}})
    out = pd.DataFrame(rows).sort_values("misfit_fit").reset_index(drop=True)
    out.to_csv(ROOT / "figures" / "paper" / "table_holdout_routing.csv", index=False)
    pd.set_option("display.width", 300)
    print("fit on Galchhi+Devghat (7 criteria), predict Betrawati/Malekhu/Kali Khola (4 criteria):")
    print(out.head(8)[["run", "misfit_fit", "pass_fit", "misfit_pred", "pass_pred", "betrawati_arr_s", "malekhu_arr_s", "kali_arr_s", "kali_rise_m"]].to_string(index=False))
    # identifiability: misfit surface over (h_bank, fp_scale) at each n_w, dep_uc = 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for ax, nw in zip(axes, sorted(out.n_w.unique())):
        sub = out[(out.n_w == nw) & (out.dep_uc == 0)]
        piv = sub.pivot(index="h_bank", columns="fp_scale", values="misfit_fit")
        im = ax.imshow(piv.values, origin="lower", cmap="viridis_r", vmin=out.misfit_fit.min(), vmax=min(out.misfit_fit.max(), 9))
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:g}" for c in piv.columns]); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{i:g}" for i in piv.index])
        ax.set_xlabel("storage width / channel width"); ax.set_ylabel("bank height above river (m)"); ax.set_title(f"$n_w$ = {nw:g}", fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w" if piv.values[i, j] > 3 else "k")
    fig.colorbar(im, ax=axes, shrink=0.85, label="misfit (Galchhi + Devghat)")
    fig.savefig(ROOT / "figures" / "paper" / "fig_routing_identifiability.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
