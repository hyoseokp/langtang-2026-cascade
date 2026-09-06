"""Rasterise satellite-mapped affected/flooded surfaces onto the simulation grid and summarise reach."""

from __future__ import annotations

import os

import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
UP = ROOT / "data"


def main(tag: str = "corridor60") -> None:
    meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    cell = meta["cell_m"]; rows, cols = meta["rows"], meta["cols"]
    left, right, bottom, top = meta["extent_utm45n"]
    transform = rasterio.transform.from_origin(left, top, cell, cell)
    route = gpd.read_file(UP / "data_processed/terrain/downstream_route_osm_candidate.geojson").to_crs(32645).geometry.iloc[0]
    center = shape(json.loads((UP / "data_processed/terrain/observed_corridor_centerline.geojson").read_text("utf-8"))["features"][0]["geometry"])

    layers = {}
    aff = gpd.read_file(UP / "data_processed/source/unosat_4260/affected_surface_utm45n.geojson").explode(index_parts=False)
    layers["unosat_affected"] = aff
    ext = gpd.read_file(UP / "data_processed/source/unosat_4260/analysis_extent_utm45n.geojson")
    for i in (1, 2, 3):
        zp = UP / "data_raw/extent" / f"emsr927_aoi0{i}_observed_event_v1.zip"
        with zipfile.ZipFile(zp) as zf:
            name = [n for n in zf.namelist() if n.endswith(".json")][0]
            g = gpd.read_file(f"zip://{zp}!{name}")
        layers[f"emsr927_aoi0{i}"] = g.to_crs(32645).explode(index_parts=False)
    zp = UP / "data_raw/extent/unosat_floodextent_20260826_nepal.zip"
    layers["unosat_floodextent"] = gpd.read_file(f"zip://{zp}!FloodExtent_20260826_Nepal.shp").to_crs(32645).explode(index_parts=False)

    summary = {}
    masks = {}
    for name, g in layers.items():
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not len(g):
            summary[name] = {"features": 0}
            continue
        mask = rasterize([(geom, 1) for geom in g.geometry], out_shape=(rows, cols), transform=transform,
                         all_touched=False, dtype=np.uint8).astype(bool)
        masks[name] = mask
        cent = g.geometry.centroid
        cr = np.array([route.project(p) for p in cent]); dr = np.array([route.distance(p) for p in cent])
        cc = np.array([center.project(p) for p in cent]); dc = np.array([center.distance(p) for p in cent])
        a = g.geometry.area.values
        near_r = dr < 2000; near_c = dc < 2000
        bins = np.arange(0, 185, 5)
        by_reach = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = near_r & (cr / 1e3 >= lo) & (cr / 1e3 < hi)
            if m.sum():
                by_reach.append({"route_km": [int(lo), int(hi)], "parts": int(m.sum()), "area_km2": float(a[m].sum() / 1e6)})
        summary[name] = {
            "features": int(len(g)), "area_km2": float(a.sum() / 1e6), "grid_cells": int(mask.sum()),
            "grid_area_km2": float(mask.sum() * cell * cell / 1e6), "bounds_utm": [float(v) for v in g.total_bounds],
            "columns": [c for c in g.columns if c != "geometry"][:12],
            "route_chainage_km_range_within_2km": [float(cr[near_r].min() / 1e3), float(cr[near_r].max() / 1e3)] if near_r.any() else None,
            "centerline_chainage_km_range_within_2km": [float(cc[near_c].min() / 1e3), float(cc[near_c].max() / 1e3)] if near_c.any() else None,
            "area_by_5km_route_reach": by_reach,
        }
        print(name, {k: v for k, v in summary[name].items() if k not in ("area_by_5km_route_reach", "columns")})
    ext_mask = rasterize([(geom, 1) for geom in ext.geometry], out_shape=(rows, cols), transform=transform, dtype=np.uint8).astype(bool)
    np.savez_compressed(ROOT / "inputs" / f"{tag}_observed_footprint.npz", analysis_extent=ext_mask,
                        **{k: v for k, v in masks.items()})
    (ROOT / "inputs" / f"{tag}_observed_footprint.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "corridor60")
