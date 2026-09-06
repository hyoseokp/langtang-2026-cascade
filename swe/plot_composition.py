"""Composition of the moving mixture versus time: stacked volume fractions of rock, ice and water (stored frames only).
The mixture is the set of wet cells with solids fraction > 0.05 (the event flow); rock = solids minus ice."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
ROCK, ICE, WATER = (0.55, 0.35, 0.15), (0.82, 0.93, 1.0), (0.10, 0.40, 0.85)
ARRIVALS = [(420, "Gyirong"), (1140, "Syabrubesi"), (3120, "Betrawati"), (5900, "Galchhi"), (16000, "Kali Khola"), (21700, "Devghat")]


def composition(run: str, cmin: float = 0.05):
    rows = []
    for fp in sorted((ROOT / "runs" / run / "frames").glob("frame_*.npz")):
        fr = np.load(fp); h = fr["h"].astype(np.float64); c = fr["c"].astype(np.float64); ice = fr["ice"].astype(np.float64)
        m = c > cmin
        sol = float((h[m] * c[m]).sum()); ic = float((h[m] * ice[m]).sum()); tot = float(h[m].sum())
        rows.append({"t": float(fr["time_s"]), "rock": sol - ic, "ice": ic, "water": tot - sol, "total": tot})
    return pd.DataFrame(rows)


def draw(ax, df, tmax_s, xlabel):
    t = df.t / 60 if tmax_s <= 7200 else df.t / 3600
    tot = df.total.replace(0, np.nan)
    fr, fi, fw = df.rock / tot, df.ice / tot, df.water / tot
    ax.fill_between(t, 0, fr, color=ROCK, lw=0, label="rock")
    ax.fill_between(t, fr, fr + fi, color=ICE, lw=0, label="ice")
    ax.fill_between(t, fr + fi, 1, color=WATER, lw=0, label="water")
    for ts, name in ARRIVALS:
        if ts <= tmax_s:
            x = ts / 60 if tmax_s <= 7200 else ts / 3600
            ax.axvline(x, color="k", lw=0.6, ls=":"); ax.text(x, 1.01, name, rotation=90, va="bottom", ha="center", fontsize=7)
    ax.set_xlim(0, tmax_s / 60 if tmax_s <= 7200 else tmax_s / 3600); ax.set_ylim(0, 1); ax.set_xlabel(xlabel); ax.set_ylabel("volume fraction of the moving mixture")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--run", default="imu60e_m0_i0.2"); p.add_argument("--hires", default="hires60")
    p.add_argument("--cmin", type=float, default=0.05); a = p.parse_args()
    df = composition(a.run, a.cmin); df.to_csv(ROOT / "figures" / "paper" / f"table_composition_{a.run}.csv", index=False)
    panels = [(df, 8 * 3600, "time after onset (h)")]
    hp = ROOT / "runs" / a.hires / "frames"
    if hp.exists():
        dh = composition(a.hires, a.cmin); dh.to_csv(ROOT / "figures" / "paper" / f"table_composition_{a.hires}.csv", index=False)
        panels.insert(0, (dh, float(dh.t.max()), "time after onset (min)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), dpi=150)
    for ax, (d, tmax, xl) in zip(np.atleast_1d(axes), panels):
        draw(ax, d, tmax, xl)
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Composition of the moving mixture (cells with solids > %g)" % a.cmin, fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_composition.png"); plt.close(fig)
    print(df.iloc[[0, 7, 19, 60, 120, 240, 480]].round(3).to_string(index=False) if len(df) > 480 else df.round(3).tail().to_string(index=False))


if __name__ == "__main__":
    main()
