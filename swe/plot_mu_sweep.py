"""Figure: first arrival at the three upper sites versus basal friction (one-at-a-time sweep) and release volume, from the stored table."""

from __future__ import annotations

import os

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
WIN = {"G": (373, 553, "Gyirong Port"), "R": (170, 770, "Rasuwagadhi"), "S": (770, 1370, "Syabrubesi")}
COL = {"G": "tab:orange", "R": "tab:green", "S": "tab:blue"}


def main() -> None:
    df = pd.read_csv(ROOT / "figures" / "paper" / "table_mu_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
    for ax, (tag, title) in zip(axes[:2], (("musw30a", "$n_w = n_d = 0.03$"), ("musw30b", "$n_w = 0.021,\\ n_d = 0.015$ (calibrated)"))):
        sub = df[df.run.str.startswith(tag)].copy(); sub["mu"] = [float(r.split("_")[-1]) for r in sub.run]; sub = sub.sort_values("mu")
        x = np.where(sub.mu > 0, sub.mu, 0.0005)
        for k in ("G", "R", "S"):
            lo, hi, name = WIN[k]
            ax.axhspan(lo, hi, color=COL[k], alpha=0.12)
            y = sub[f"{k}_water_arr"].to_numpy(); miss = ~np.isfinite(y)
            ax.plot(x[~miss], y[~miss], "o-", color=COL[k], label=name, ms=5)
            if miss.any():
                ax.plot(x[miss], np.full(miss.sum(), 3500), "x", color=COL[k], ms=7)
        ax.set_xscale("log"); ax.set_xticks([0.0005, 0.002, 0.005, 0.01, 0.02]); ax.set_xticklabels(["0", "0.002", "0.005", "0.01", "0.02"])
        ax.set_xlabel(r"basal friction coefficient $\mu_s$"); ax.set_ylabel("first arrival (s after onset)"); ax.set_ylim(0, 3700); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    ax = axes[2]
    vol = df[df.run.str.startswith("vol") | (df.run == "musw30a_0") | (df.run == "musw30a_0.02")].copy()
    lab = {"musw30a_0": "35 Mm3, mu 0", "musw30a_0.02": "35 Mm3, mu 0.02", "vol100_mu0": "100 Mm3, mu 0", "vol100_mu02": "100 Mm3, mu 0.02", "vol250_mu02": "250 Mm3, mu 0.02", "vol250_mu05": "250 Mm3, mu 0.05"}
    vol["label"] = [lab.get(r, r) for r in vol.run]; vol = vol.set_index("label").reindex([v for v in lab.values() if v in set(vol.label)])
    ypos = np.arange(len(vol))
    for k in ("G", "R", "S"):
        lo, hi, name = WIN[k]; ax.axvspan(lo, hi, color=COL[k], alpha=0.12)
        y = vol[f"{k}_water_arr"].to_numpy(); miss = ~np.isfinite(y)
        ax.plot(y[~miss], ypos[~miss], "o", color=COL[k], ms=5); ax.plot(np.full(miss.sum(), 3500), ypos[miss], "x", color=COL[k], ms=7)
    ax.set_yticks(ypos); ax.set_yticklabels(vol.index, fontsize=8); ax.set_xlim(0, 3700); ax.set_xlabel("first arrival (s after onset)"); ax.set_title("release volume and friction", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_mu_sweep.png"); plt.close(fig)
    print(df[["run", "G_water_arr", "R_water_arr", "S_water_arr"]].to_string(index=False))


if __name__ == "__main__":
    main()
