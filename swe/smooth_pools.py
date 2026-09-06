"""Remove the pool-and-step staircase of the conditioned DSM along the main stem: from the source down the D8 path,
raise the bed so that every cell descends by at least a fraction of the local reach slope (pools are filled up to a ramp
that meets the next step). Neighbouring channel cells are raised to the same ramp. Writes <tag>p with the grid unchanged.
Usage: python swe/smooth_pools.py --tag corridor60s [--frac 0.5] [--window-m 1000]"""

from __future__ import annotations

import os

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--tag", required=True); p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--window-m", type=float, default=1000.0); p.add_argument("--min-slope", type=float, default=5e-4)
    a = p.parse_args()
    inp = dict(np.load(ROOT / "inputs" / f"{a.tag}.npz")); meta = json.loads((ROOT / "inputs" / f"{a.tag}.json").read_text("utf-8"))
    z = inp["z"].astype(np.float64); rows, cols = z.shape; cell = meta["cell_m"]
    rcv = inp["rcv"].astype(np.int64).ravel(); src = inp["source_mask"]; channel = inp["channel"].ravel()
    r0, c0 = np.argwhere(src)[np.argmin(z[src])]; node = int(r0 * cols + c0); path = [node]
    while rcv[node] >= 0:
        node = int(rcv[node]); path.append(node)
    path = np.array(path); pr, pc = np.divmod(path, cols)
    step = np.where((np.diff(pr) != 0) & (np.diff(pc) != 0), cell * np.sqrt(2.0), cell); chain = np.r_[0.0, np.cumsum(step)]
    zf = z.ravel(); zp = zf[path].copy(); n = len(path)
    # local reach slope from a window of the (already monotone) path profile
    w = a.window_m; s_loc = np.empty(n)
    for i in range(n):
        lo = np.searchsorted(chain, chain[i] - w); hi = min(np.searchsorted(chain, chain[i] + w), n - 1)
        s_loc[i] = max((zp[lo] - zp[hi]) / max(chain[hi] - chain[lo], cell), a.min_slope)
    s_req = np.maximum(a.frac * s_loc, a.min_slope)
    ramp = zp.copy()
    for i in range(n - 2, -1, -1):  # downstream -> upstream: fill pools so the bed keeps descending into the next step
        ramp[i] = max(ramp[i], ramp[i + 1] + s_req[i] * step[i])
    raise_m = ramp - zp
    znew = zf.copy(); znew[path] = np.maximum(znew[path], ramp)
    # neighbouring channel cells (the widened main stem) follow the ramp
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nb = np.clip(pr + dr, 0, rows - 1) * cols + np.clip(pc + dc, 0, cols - 1)
            sel = channel[nb] & (znew[nb] < ramp); znew[nb[sel]] = ramp[sel]
    changed = int((znew != zf).sum()); inp["z"] = znew.reshape(rows, cols).astype(inp["z"].dtype)
    out = a.tag + "p"; np.savez_compressed(ROOT / "inputs" / f"{out}.npz", **inp)
    meta["pool_smoothing"] = {"from_tag": a.tag, "frac_of_local_slope": a.frac, "window_m": w, "min_slope": a.min_slope, "path_cells": int(n),
                              "cells_raised": changed, "raise_max_m": float(raise_m.max()), "raise_mean_m": float(raise_m.mean()),
                              "raise_volume_m3": float((znew - zf).sum() * cell * cell)}
    (ROOT / "inputs" / f"{out}.json").write_text(json.dumps(meta, indent=1), "utf-8")
    for suf in ("_transects.json", "_observed_footprint.npz", "_observed_footprint.json", "_trimline_profile.csv"):
        if (ROOT / "inputs" / f"{a.tag}{suf}").exists():
            shutil.copy2(ROOT / "inputs" / f"{a.tag}{suf}", ROOT / "inputs" / f"{out}{suf}")
    print(json.dumps(meta["pool_smoothing"], indent=1))
    big = np.nonzero(raise_m > 2)[0]
    print("segments raised > 2 m (km):", np.round(chain[big] / 1e3, 1)[::max(1, len(big) // 30)])


if __name__ == "__main__":
    main()
