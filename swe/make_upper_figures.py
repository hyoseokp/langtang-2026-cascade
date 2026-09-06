"""Upper-corridor paper figures: flow snapshots (Fig. 3) and the rheology/volume sensitivity chart (Fig. 3b)."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
DEBRIS = np.array([0.80, 0.30, 0.02]); WATER = np.array([0.05, 0.40, 0.80])


def snapshots(run: str, base: str, tag: str, times: list[float], out: Path) -> None:
    inp = np.load(ROOT / "inputs" / f"{tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    obs = np.load(ROOT / "inputs" / f"{tag}_observed_footprint.npz")["unosat_affected"]
    z = inp["z_raw"]; l, r, b, t = meta["extent_utm45n"]; cell = meta["cell_m"]
    shade = LightSource(315, 45).shade(z, cmap=plt.get_cmap("gray"), vert_exag=1, dx=cell, dy=cell,
                                       vmin=np.percentile(z, 1), vmax=np.percentile(z, 99.5), blend_mode="soft")[..., :3]
    hb = np.load(ROOT / "runs" / base / "final_state.npz")["h"].ravel()
    frames = sorted((ROOT / "runs" / run / "frames").glob("frame_*.npz"))
    ftimes = np.array([float(np.load(f)["time_s"]) for f in frames])
    fig, axes = plt.subplots(2, 2, figsize=(13, 13), dpi=130)
    xs = np.linspace(l, r, z.shape[1]); ys = np.linspace(t, b, z.shape[0])
    for ax, tt in zip(axes.ravel(), times):
        fr = np.load(frames[int(np.argmin(np.abs(ftimes - tt)))])
        h = np.zeros(z.size, np.float32); c = np.zeros(z.size, np.float32)
        h[fr["idx"]] = fr["h"]; c[fr["idx"]] = fr["c"]
        excess = np.maximum(h - hb, 0).reshape(z.shape); c = c.reshape(z.shape); h = h.reshape(z.shape)
        sig = np.maximum(excess, h * c)
        rgba = np.zeros(z.shape + (4,), np.float32)
        rgba[..., :3] = WATER * (1 - c[..., None]) + DEBRIS * c[..., None]
        rgba[..., 3] = np.where(sig > 0.3, 0.3 + 0.7 * (1 - np.exp(-sig / 5)), 0)
        ax.imshow(shade, extent=(l, r, b, t)); ax.imshow(rgba, extent=(l, r, b, t), interpolation="nearest")
        ax.contour(xs, ys, obs.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=0.5)
        for k, s in meta["stations"].items():
            ax.plot(s["x"], s["y"], "w^", markeredgecolor="k", markersize=7)
            ax.annotate(s["label"], (s["x"], s["y"]), xytext=(5, 5), textcoords="offset points", fontsize=7, color="w")
        ax.set_title(f"t = {float(fr['time_s']):.0f} s  |  max depth {h.max():.0f} m", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(handles=[Line2D([0], [0], color=DEBRIS, lw=6, label="debris-rich mixture"), Line2D([0], [0], color=WATER, lw=6, label="water-rich"),
                               Line2D([0], [0], color="#00e5ff", lw=1, label="UNOSAT affected surface")], loc="lower left", fontsize=8)
    fig.suptitle(f"Upper corridor ({tag}, run {run}): flow depth and composition", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "fig_upper_snapshots.png"); plt.close(fig)


def sensitivity(rows: list[dict], out: Path) -> None:
    df = pd.DataFrame(rows); df.to_csv(out / "table_upper_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    y = np.arange(len(df))
    for key, lab, col, win in (("G", "Gyirong", "tab:orange", (373, 553)), ("R", "Rasuwagadhi", "tab:green", (170, 770)), ("S", "Syabrubesi", "tab:blue", (770, 1370))):
        ax.axvspan(win[0], win[1], color=col, alpha=0.12)
        ax.scatter(df[key], y, color=col, label=f"{lab} (window shaded)", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(df.label, fontsize=8); ax.set_xlabel("modelled arrival (s after onset)")
    ax.set_xlim(0, 3000); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_upper_sensitivity.png"); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--run", default="u30f_mu0"); p.add_argument("--base", default="u30f_spinup")
    p.add_argument("--tag", default="upper30f"); a = p.parse_args()
    out = ROOT / "figures" / "paper"; out.mkdir(parents=True, exist_ok=True)
    snapshots(a.run, a.base, a.tag, [120, 420, 780, 1260], out)
    rows = []
    for run, label in [("u30_dry_mu08_n06", "dry, mu 0.08 n 0.06"), ("u30_dry_mu04_n03", "dry, mu 0.04 n 0.03"), ("u30_dry_mu02_n03", "dry, mu 0.02 n 0.03"),
                       ("u30_mix_mu02_n03", "river, mu 0.02 n 0.03"), ("u30_src50_mu04_n03", "50% source water, mu 0.04"), ("u30_v20_c50_e0", "20 Mm3, 50% water, mu 0.02"),
                       ("u30_melt_tau600", "forced melt tau 600 s, mu 0.02"), ("u30h_mu005", "mu 0.005, ice 25%, entrainment"), ("u30h_mu0", "mu 0, ice 25%, entrainment"),
                       ("u30f_mu0", "mu 0, final conditioning")]:
        path = ROOT / "runs" / run / "series.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path); tt = s.time_s.to_numpy()
        def arr(st):
            hm = s[f"{st}_hmax"].to_numpy(); i = np.nonzero(hm - hm[0] > 0.5)[0]; return float(tt[i[0]]) if len(i) else np.nan
        rows.append({"run": run, "label": label, "G": arr("gyirong_cctv_arrival"), "R": arr("rasuwagadhi_signal_loss"), "S": arr("syabrubesi_signal_loss")})
    sensitivity(rows, out)
    print("upper figures written")


if __name__ == "__main__":
    main()
