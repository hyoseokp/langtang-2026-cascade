"""Sweep figure from the stored score table only: misfit vs parameter, and the criterion matrix of the best runs."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
PARAMS = [("n_w", "Manning $n_w$ (water)", False), ("n_d", "Manning $n_d$ (debris, $c=1$)", False),
          ("release_c0", "release solid fraction $c_0$", False), ("dep_uc", "settling threshold $u_{dep}$ (m/s)", False),
          ("dep_tau", r"settling time $\tau_{dep}$ (s)", True), ("erosion_k", "entrainment $K$", True)]
CRIT = [("gyirong_arr_s", "Gyirong arrival", 463, 90), ("rasuwagadhi_arr_s", "Rasuwagadhi arrival", 470, 300),
        ("syabrubesi_arr_s", "Syabrubesi arrival", 1070, 300), ("betrawati_arr_s", "Betrawati arrival", 2870, 300),
        ("galchhi_arr_s", "Galchhi rise", 6840, 1080), ("galchhi_peak_s", "Galchhi peak time", 8100, 1260),
        ("galchhi_rise_m", "Galchhi rise (m)", 8.8, 1.0), ("malekhu_arr_s", "Malekhu arrival", 9840, 300),
        ("kali_arr_s", "Kali Khola rise", 17460, 1260), ("kali_rise_m", "Kali Khola rise (m)", 5.6, 0.7),
        ("devghat_arr_s", "Devghat rise", 21510, 300), ("devghat_peak_s", "Devghat peak time", 26640, 900),
        ("devghat_rise_m", "Devghat rise (m)", 1.9, 0.4), ("devghat_ratio", "Devghat peak ratio", 2.25, 0.55),
        ("devghat_excess_Mm3", "Devghat excess (Mm$^3$)", 20.0, 10.0)]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--prefix", default="sw1"); p.add_argument("--top", type=int, default=8)
    a = p.parse_args()
    df = pd.read_csv(ROOT / "sweep" / f"{a.prefix}_scores.csv").sort_values("misfit").reset_index(drop=True)
    out = ROOT / "figures" / "paper"
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), dpi=150)
    for ax, (key, label, lg) in zip(axes.ravel(), PARAMS):
        ax.scatter(df[key], df.misfit, c=df.n_pass, cmap="viridis", s=28, edgecolor="k", linewidth=0.3)
        ax.scatter(df[key].iloc[0], df.misfit.iloc[0], s=140, facecolor="none", edgecolor="red", linewidth=1.5)
        if lg:
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(label); ax.set_ylabel("misfit"); ax.grid(alpha=0.3)
    sc = axes[0, 0].collections[0]; cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02); cb.set_label("criteria passed (of %d)" % len(CRIT))
    fig.suptitle(f"Inverse sweep ({len(df)} runs): misfit vs parameter, best run circled", fontsize=10)
    fig.savefig(out / f"fig_sweep_{a.prefix}.png", bbox_inches="tight"); plt.close(fig)
    # criterion matrix for the top runs: normalised deviation (model - target)/tol
    top = df.head(a.top)
    M = np.array([[(row[k] - tgt) / tol if np.isfinite(row[k]) else np.nan for k, _, tgt, tol in CRIT] for _, row in top.iterrows()])
    fig, ax = plt.subplots(figsize=(10, 0.45 * a.top + 2.2), dpi=150)
    im = ax.imshow(np.clip(M, -3, 3), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "miss" if not np.isfinite(v) else f"{v:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if (np.isfinite(v) and abs(v) > 1.8) else "black")
    ax.set_xticks(range(len(CRIT))); ax.set_xticklabels([c[1] for c in CRIT], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{r.run}  (misfit {r.misfit:.2f}, {int(r.n_pass)} pass)" for r in top.itertuples()], fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("(model - observed) / tolerance")
    fig.tight_layout(); fig.savefig(out / f"fig_sweep_matrix_{a.prefix}.png"); plt.close(fig)
    print(df.head(a.top)[["run", "misfit", "n_pass"] + [k for k, _, _ in PARAMS]].to_string(index=False))


if __name__ == "__main__":
    main()
