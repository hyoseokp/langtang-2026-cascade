"""Valley-floor conditioning: lower DSM cells on the satellite-mapped valley floor to a gentle cross-slope.

Only the WIDTH information of the mapped affected/flooded surface is used; cells are never raised.
Writes a new input tag (<tag>v) with identical grid, so footprint/trimline products carry over.
"""

from __future__ import annotations

import os

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
HALFWIDTH_M = 150.0
BASE_OFFSET_M = 2.0
CROSS_SLOPE = 0.02
MAX_WALL_M = 60.0
MAX_LOWER_M = 15.0  # typical DSM canopy/terrace bias; valley walls are never cut deeper than this


def main(tag: str) -> None:
    inp = dict(np.load(ROOT / "inputs" / f"{tag}.npz"))
    meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    obs = np.load(ROOT / "inputs" / f"{tag}_observed_footprint.npz")
    cell = meta["cell_m"]
    z = inp["z"].astype(np.float64); channel = inp["channel"]
    k = int(round(HALFWIDTH_M / cell))
    size = 2 * k + 1
    zch = np.where(channel, z, np.inf)
    zmin = ndimage.minimum_filter(zch, size=size, mode="nearest")
    dist = ndimage.distance_transform_edt(~channel) * cell
    mapped = obs["unosat_affected"] | obs["unosat_floodextent"]
    floor = mapped & (dist <= HALFWIDTH_M) & ~channel & (z - zmin < MAX_WALL_M) & np.isfinite(zmin)
    target = zmin + BASE_OFFSET_M + CROSS_SLOPE * dist
    z_new = np.where(floor & (z > target), np.maximum(target, z - MAX_LOWER_M), z)
    lowered = z - z_new
    n = int((lowered > 0.01).sum())
    print(f"{tag}: valley-floor cells {int(floor.sum())}, lowered {n}, mean {lowered[lowered>0.01].mean():.1f} m, "
          f"max {lowered.max():.1f} m, volume {lowered.sum()*cell*cell/1e6:.1f} Mm3")
    inp["z"] = z_new.astype(np.float32)
    inp["valley_lowering"] = lowered.astype(np.float32)
    new = f"{tag}v"
    np.savez_compressed(ROOT / "inputs" / f"{new}.npz", **inp)
    meta = dict(meta); meta["tag"] = new
    meta["valley_conditioning"] = {"source": "UNOSAT affected surface + flood extent (width only)", "halfwidth_m": HALFWIDTH_M,
                                   "base_offset_m": BASE_OFFSET_M, "cross_slope": CROSS_SLOPE, "max_wall_m": MAX_WALL_M,
                                   "max_lower_m": MAX_LOWER_M,
                                   "cells_lowered": n, "volume_lowered_m3": float(lowered.sum() * cell * cell)}
    (ROOT / "inputs" / f"{new}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for suffix in ("_observed_footprint.npz", "_observed_footprint.json", "_trimline_profile.csv", "_geopera_dh.npz"):
        src = ROOT / "inputs" / f"{tag}{suffix}"
        if src.exists():
            shutil.copy(src, ROOT / "inputs" / f"{new}{suffix}")
    print("wrote", new)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "upper30")
