"""Maximum flow height along the corridor from satellite-mapped affected surfaces (trimline proxy).

For each 500 m chainage bin: trimline height = (highest DEM elevation inside the mapped affected
surface within the bin) - (thalweg elevation of the bin). Uses the raw 30 m DEM on the simulation grid.
"""

from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
UP = ROOT / "data"
PLANET = UP / "data_processed/disturbance/planet_preliminary_class_probabilities_12m.tif"


def main(tag: str = "corridor60", bin_m: float = 500.0) -> None:
    inp = np.load(ROOT / "inputs" / f"{tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    obs = np.load(ROOT / "inputs" / f"{tag}_observed_footprint.npz")
    z = inp["z_raw"]; cell = meta["cell_m"]; left, right, bottom, top = meta["extent_utm45n"]
    transform = rasterio.transform.from_origin(left, top, cell, cell)
    masks = {"unosat_affected": obs["unosat_affected"], "unosat_floodextent": obs["unosat_floodextent"]}
    if PLANET.exists():
        with rasterio.open(PLANET) as src:
            probs = src.read()
            dst = np.zeros((probs.shape[0],) + z.shape, np.float32)
            for b in range(probs.shape[0]):
                reproject(probs[b], dst[b], src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                          dst_crs="EPSG:32645", resampling=Resampling.average, src_nodata=src.nodata, dst_nodata=0)
        affected = dst[1] + dst[2] + dst[3]  # stripping_scour + fresh_deposit + water
        masks["planet_disturbed_p50"] = affected > 0.5
    from scipy import ndimage
    edges_of = {name: m & ~ndimage.binary_erosion(m) for name, m in masks.items()}
    rows = []
    for line_key in ("route_chainage_m", "upper_chainage_m"):
        ch = inp[line_key]
        finite = np.isfinite(ch)
        edges = np.arange(0, np.nanmax(ch) + bin_m, bin_m)
        for lo, hi in zip(edges[:-1], edges[1:]):
            inbin = finite & (ch >= lo) & (ch < hi)
            if not inbin.any():
                continue
            thalweg = float(z[inbin].min())
            row = {"line": line_key.replace("_chainage_m", ""), "chainage_km": (lo + hi) / 2e3, "thalweg_z": thalweg}
            for name, m in masks.items():
                sel = inbin & m
                row[f"{name}_cells"] = int(sel.sum())
                edge = inbin & edges_of[name]
                row[f"{name}_trimline_m"] = float(np.median(z[edge]) - thalweg) if edge.any() else np.nan
                row[f"{name}_trimline_p25_m"] = float(np.percentile(z[edge], 25) - thalweg) if edge.any() else np.nan
                row[f"{name}_trimline_p75_m"] = float(np.percentile(z[edge], 75) - thalweg) if edge.any() else np.nan
                row[f"{name}_width_m"] = float(sel.sum() * cell * cell / bin_m)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / "inputs" / f"{tag}_trimline_profile.csv"
    df.to_csv(out, index=False)
    for line in ("route", "upper"):
        d = df[df.line == line]
        print(line, "bins", len(d))
        for name in masks:
            v = d[f"{name}_trimline_m"].dropna()
            if len(v):
                print(f"  {name}: trimline height median {v.median():.1f} m, p90 {v.quantile(0.9):.1f} m, max {v.max():.1f} m over {len(v)} bins; "
                      f"width median {d[f'{name}_width_m'].replace(0, np.nan).median():.0f} m")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "corridor60")
