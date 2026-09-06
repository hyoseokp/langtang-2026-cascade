"""3x3 figure from stored clean animation panels: rows = views (oblique 3-D, top view, longitudinal profile), columns = times,
with one shared legend below. Usage: python swe/make_fig_hires_grid.py --run hiresN --frame-dt 10 --frames 40 120 360"""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
DEBRIS = (0.80, 0.30, 0.02); WATER = (0.05, 0.40, 0.80); BASE = (0.55, 0.65, 0.75); DEPOSIT = (0.3, 0.16, 0.05)


def autocrop(im: Image.Image, pad: int = 10) -> Image.Image:
    a = np.asarray(im.convert("L")); ys, xs = np.nonzero(a < 250)
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height)))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--run", default="hiresN"); p.add_argument("--frame-dt", type=float, default=10.0)
    p.add_argument("--frames", type=int, nargs=3, default=[40, 120, 360]); p.add_argument("--out", default="fig_upper_grid.png")
    a = p.parse_args()
    base = ROOT / "figures" / "animations" / a.run / "paper"
    views = [("upper_oblique_{:05d}.png", "oblique view"), ("upper_top_{:05d}.png", "top view"), ("p_{:05d}.png", "longitudinal profile")]
    labels = [f"{f * a.frame_dt:.0f} s" for f in a.frames]
    ims = [[autocrop(Image.open(base / pat.format(f))) for f in a.frames] for pat, _ in views]
    ratios = [max(im.height / im.width for im in row) for row in ims]
    fig = plt.figure(figsize=(18, 6 * sum(ratios) + 1.6), dpi=170)
    gs = fig.add_gridspec(4, 3, height_ratios=ratios + [0.16], hspace=0.05, wspace=0.03, left=0.01, right=0.99, top=0.985, bottom=0.005)
    k = 0
    for i, ((_, name), row) in enumerate(zip(views, ims)):
        for j, im in enumerate(row):
            ax = fig.add_subplot(gs[i, j]); ax.imshow(im); ax.set_axis_off()
            ax.text(0.01, 0.99, f"{'abcdefghi'[k]}", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="left")
            ax.text(0.5, 0.99, f"{name}, t = {labels[j]}", transform=ax.transAxes, fontsize=17, va="top", ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3)); k += 1
    lax = fig.add_subplot(gs[3, :]); lax.set_axis_off()
    handles = [Patch(facecolor=DEBRIS, label="debris flow (solids ≥ 35 %)"), Patch(facecolor=WATER, label="flood water (solids → 0)"),
               Patch(facecolor=BASE, label="pre-event river"), Line2D([0], [0], color="#00e5ff", lw=2.5, label="UNOSAT mapped affected surface"),
               Patch(facecolor=(0.6, 0.6, 0.6), label="profile: rise above the pre-event level"), Patch(facecolor=DEBRIS, label="profile: solids in the flow, h·c"),
               Patch(facecolor=DEPOSIT, hatch="///", edgecolor="k", label="profile: settled deposit"), Line2D([0], [0], color=(0.3, 0.2, 0.1), lw=1.5, label="profile: channel scour")]
    lax.legend(handles=handles, loc="center", ncol=4, fontsize=16, frameon=False, handlelength=2.2, columnspacing=2.0)
    fig.savefig(ROOT / "figures" / "paper" / a.out); print(ROOT / "figures" / "paper" / a.out)


if __name__ == "__main__":
    main()
