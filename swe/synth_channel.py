"""Replace the DSM main-stem channel below a start station with a hydraulic-geometry channel.

The longitudinal profile is the DSM thalweg along the D8 main stem, median-filtered over
SMOOTH_M and made monotone by isotonic (pool-adjacent-violators) regression; the width is a
function of contributing area anchored on mapped pre-event water widths. Cells within W/2 of
the path are set to the profile with a small cross slope; near-bank voids are raised.
Writes <tag>s.npz/json. Only the reach downstream of START_STATION is changed.
"""

from __future__ import annotations

import os

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))


def width_from_area(acc_km2: np.ndarray) -> np.ndarray:
    # anchors: gorge above Betrawati ~80 m; Galchhi (2900 km2) 115 m; Mugling (~6000) 200 m; Devghat (~9000+) 400 m
    import os
    scale = float(os.environ.get("LOWER_WIDTH_SCALE", "1.0"))
    pts = np.array([[50, 60], [500, 80], [2900, 115], [6000, 200 * scale], [9000, 400 * scale], [20000, 450 * scale]], float)
    return np.interp(np.log(np.clip(acc_km2, 1, None)), np.log(pts[:, 0]), pts[:, 1])


def isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a non-increasing fit."""
    y = -y.astype(float)
    n = len(y); blocks = [[y[i], 1, i, i] for i in range(n)]  # mean, count, start, end
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            a = out.pop(); c = out.pop()
            m = (a[0] * a[1] + c[0] * c[1]) / (a[1] + c[1])
            out.append([m, a[1] + c[1], c[2], a[3]])
    fit = np.empty(n)
    for m, cnt, s, e in out:
        fit[s:e + 1] = m
    return -fit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="corridor60h"); p.add_argument("--out-tag", default="corridor60s")
    p.add_argument("--start", default="syabrubesi_signal_loss"); p.add_argument("--smooth-m", type=float, default=5000.0)
    p.add_argument("--cross-slope", type=float, default=0.01)
    a = p.parse_args()
    d = dict(np.load(ROOT / "inputs" / f"{a.tag}.npz")); meta = json.loads((ROOT / "inputs" / f"{a.tag}.json").read_text("utf-8"))
    z = d["z"].astype(np.float64); zr = d["z_raw"].astype(np.float64); rows, cols = z.shape; cell = meta["cell_m"]
    rcv = d["rcv"].astype(np.int64).ravel(); acc = d["acc_cells"].astype(np.float64) * cell * cell / 1e6
    st = meta["stations"][a.start]
    node = st["row"] * cols + st["col"]; path = [node]
    while rcv[node] >= 0:
        node = int(rcv[node]); path.append(node)
    path = np.array(path); pr, pc = np.divmod(path, cols)
    step = np.where((np.diff(pr) != 0) & (np.diff(pc) != 0), cell * np.sqrt(2), cell); chain = np.r_[0.0, np.cumsum(step)]
    # thalweg from the raw DSM (3x3 min), median filter over smooth_m, isotonic decreasing fit
    nb = np.stack([zr[np.clip(pr + dr, 0, rows - 1), np.clip(pc + dc, 0, cols - 1)] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], 1)
    thal = nb.min(1)
    k = max(3, int(a.smooth_m / cell) | 1)
    med = ndimage.median_filter(thal, size=k, mode="nearest")
    prof = isotonic_decreasing(med)
    prof = np.minimum(prof, prof[0])  # never above the start station bed
    prof = np.minimum(prof, z[st["row"], st["col"]])  # and never above the conditioned bed at the junction (else a dam)
    width = width_from_area(acc[pr, pc])
    # rasterise: nearest path cell for every grid cell, distance from path
    pathmask = np.zeros(z.shape, bool); pathmask[pr, pc] = True
    dist, (ir, ic) = ndimage.distance_transform_edt(~pathmask, return_indices=True)
    dist = dist * cell
    idx_of = np.full(z.shape, -1, np.int64); idx_of[pr, pc] = np.arange(len(path))
    near = idx_of[ir, ic]
    W = width[near]; Zp = prof[near]
    inside = dist <= W / 2
    bank = (dist > W / 2) & (dist <= W)  # void guard next to the channel
    z_new = z.copy()
    z_new[inside] = Zp[inside] + a.cross_slope * dist[inside]
    z_new[bank] = np.maximum(z[bank], Zp[bank] + 0.5 * a.cross_slope * W[bank] + 1.0)
    # keep the upper corridor and everything not adjacent to the main stem unchanged
    changed = inside | bank
    d["z"] = z_new.astype(np.float32); d["synth_channel"] = inside
    d["synth_channel_mask"] = inside  # erodible zone stays the original channel network
    np.savez_compressed(ROOT / "inputs" / f"{a.out_tag}.npz", **d)
    meta = dict(meta); meta["tag"] = a.out_tag
    meta["synthetic_channel"] = {"start_station": a.start, "path_cells": int(len(path)), "length_km": float(chain[-1] / 1e3),
                                 "smooth_m": a.smooth_m, "width_anchors_km2_m": [[50, 60], [500, 80], [2900, 115], [6000, 200], [9000, 400]],
                                 "cells_changed": int(changed.sum()), "profile_start_m": float(prof[0]), "profile_end_m": float(prof[-1]),
                                 "raw_thalweg_rms_dev_m": float(np.sqrt(np.mean((thal - prof) ** 2)))}
    (ROOT / "inputs" / f"{a.out_tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for suffix in ("_observed_footprint.npz", "_observed_footprint.json", "_trimline_profile.csv"):
        src = ROOT / "inputs" / f"{a.tag}{suffix}"
        if src.exists():
            shutil.copy(src, ROOT / "inputs" / f"{a.out_tag}{suffix}")
    print(f"{a.out_tag}: path {len(path)} cells, {chain[-1]/1e3:.1f} km, profile {prof[0]:.0f}->{prof[-1]:.0f} m, "
          f"slope {(prof[0]-prof[-1])/chain[-1]*100:.2f}%, width {width.min():.0f}-{width.max():.0f} m, changed {changed.sum()} cells, "
          f"thalweg rms dev {np.sqrt(np.mean((thal-prof)**2)):.1f} m")


if __name__ == "__main__":
    main()
