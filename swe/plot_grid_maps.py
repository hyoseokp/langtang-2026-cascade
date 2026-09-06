"""Misfit colormaps over structured parameter grids (stored series only):
upper corridor: (mu_s, n) grid scored on the three timing windows; the downstream (bank height, storage width) map is made by holdout_routing.py."""

from __future__ import annotations

import os

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
WIN = {"gyirong_cctv_arrival": (463, 90), "rasuwagadhi_signal_loss": (470, 300), "syabrubesi_signal_loss": (1070, 300)}
import sys
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "grid30"
MUS = [0, 0.002, 0.005, 0.01, 0.02]; NS = [0.02, 0.025, 0.03, 0.035]


def arrival(s, st):
    q = np.abs(s[f"{st}_Q"].to_numpy()); stg = s[f"{st}_stage"].to_numpy(); t = s.time_s.to_numpy()
    i = np.nonzero((stg - stg[0] > 0.5) | (q > 2 * q[0] + 50))[0]
    return float(t[i[0]]) if len(i) else np.nan


def main() -> None:
    M = np.full((len(MUS), len(NS)), np.nan); P = np.zeros_like(M); A = {}
    for i, mu in enumerate(MUS):
        for j, n in enumerate(NS):
            run = f"{PREFIX}_m{mu}_n{n}"
            p = ROOT / "runs" / run / "series.csv"
            if not p.exists():
                continue
            s = pd.read_csv(p); terms = []; arr = {}
            for st, (tgt, tol) in WIN.items():
                v = arrival(s, st); arr[st] = v; terms.append(9.0 if not np.isfinite(v) else min(((v - tgt) / tol) ** 2, 9.0))
            M[i, j] = np.mean(terms); P[i, j] = sum(t <= 1 for t in terms); A[run] = arr
    pd.DataFrame([{"run": k, **v} for k, v in A.items()]).to_csv(ROOT / "figures" / "paper" / f"table_{PREFIX}.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4), dpi=150)
    im = ax.imshow(M, origin="lower", cmap="viridis_r", vmin=0, vmax=min(np.nanmax(M), 9))
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"{n:g}" for n in NS]); ax.set_yticks(range(len(MUS))); ax.set_yticklabels([f"{m:g}" for m in MUS])
    ax.set_xlabel("Manning roughness $n$ (water = debris)"); ax.set_ylabel(r"basal friction $\mu_s$")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}\n{int(P[i, j])}/3", ha="center", va="center", fontsize=7, color="w" if M[i, j] > 3 else "k")
    fig.colorbar(im, ax=ax, label="timing misfit (Gyirong, Rasuwagadhi, Syabrubesi)")
    ax.set_title("Upper corridor, 30 m: misfit over the ($\\mu_s$, $n$) grid; cells show misfit and windows met", fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / (f"fig_grid_mu_n.png" if PREFIX == "grid30e" else f"fig_grid_mu_n_{PREFIX}.png")); plt.close(fig)
    print(pd.DataFrame(M, index=MUS, columns=NS).round(2))


if __name__ == "__main__":
    main()
