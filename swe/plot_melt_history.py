"""Melt history of the release ice for the energy-conserving runs at mu = 0 and each ice fraction (stored series only)."""

from __future__ import annotations

import os

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
V_REL = 35.4e6


def main() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ice in (0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8):
        p = ROOT / "runs" / f"imu60e_m0_i{ice:g}" / "series.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p); t = s.time_s.to_numpy(); m = s.melted_m3.to_numpy(); ice0 = V_REL * ice
        ax[0].plot(t / 60, m / 1e6, label=f"ice {ice:g}"); ax[1].plot(t / 60, 100 * m / ice0, label=f"ice {ice:g}")
    p = ROOT / "runs" / "me30" / "series.csv"
    if p.exists():
        s = pd.read_csv(p); ax[0].plot(s.time_s / 60, s.melted_m3 / 1e6, "k--", lw=1, label="30 m grid, ice 0.25"); ax[1].plot(s.time_s / 60, 100 * s.melted_m3 / (0.25 * V_REL), "k--", lw=1)
    p = ROOT / "runs" / "prod30" / "series.csv"
    if p.exists():
        s = pd.read_csv(p); ax[1].plot(s.time_s / 60, 100 * s.melted_m3 / (0.25 * V_REL), "k:", lw=1, label="friction heat only (30 m, ice 0.25)")
    for a in ax:
        a.set_xlim(0, 120); a.grid(alpha=0.3); a.set_xlabel("time after onset (min)")
        for x, lab in ((7, "Gyirong"), (8, ""), (19, "Syabrubesi"), (50, "Betrawati")):
            a.axvline(x, color="grey", lw=0.6, ls=":")
    ax[0].set_ylabel("melted ice (10$^6$ m$^3$)"); ax[1].set_ylabel("fraction of the release ice melted (%)"); ax[1].set_ylim(0, 100)
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_melt_history.png"); plt.close(fig)
    print("written")


if __name__ == "__main__":
    main()
