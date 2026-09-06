"""Build the corridor grid, drainage, source, inflow and station inputs for the SWE run.

Independent of the r.avaflow pipeline: only raw/processed terrain, the UNOSAT
source polygon, the mapped downstream route and the observation registry are read.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import LineString, shape

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
UPSTREAM = ROOT / "data"
OUT = ROOT / "inputs"
DEM_PATH = UPSTREAM / "data_processed/terrain/full_reconstruction_utm45n_egm2008.tif"
SOURCE_PATH = UPSTREAM / "data_processed/source/unosat_4260/detachment_zone_utm45n.geojson"
ROUTE_PATH = UPSTREAM / "data_processed/terrain/downstream_route_osm_candidate.geojson"
CENTERLINE_PATH = UPSTREAM / "data_processed/terrain/observed_corridor_centerline.geojson"
OBS_PATH = UPSTREAM / "data_raw/observations/event_observations.csv"
OFFSETS_PATH = UPSTREAM / "data_processed/terrain/downstream_route_gauge_offsets.csv"

SOURCE_VOLUME_M3 = 35_420_000.0  # QHNU conditional estimate used by the upstream project

# Conditional monsoon baseflow assumptions (m3/s). Not measured; documented in inputs.json.
EXTERNAL_INFLOWS = {
    "bhote_koshi_tibet": {"anchor_lonlat": [85.372, 28.303], "snap_m": 800.0, "q_m3_s": 150.0},  # Timure reach
    # interior anchors on the named rivers (Arughat Bazaar, Dumre); snapped to the channel cell
    "budhi_gandaki": {"anchor_lonlat": [84.817, 28.045], "snap_m": 1500.0, "q_m3_s": 500.0},
    "marsyangdi": {"anchor_lonlat": [84.418, 27.977], "snap_m": 1500.0, "q_m3_s": 700.0},
}
# Kali Gandaki / Seti enter below the Devghat candidate cell and are therefore not injected.
INFLOW_INSET_CELLS = 4
MIN_SLOPE = 5e-4
DEEP_PIT_M = float(__import__("os").environ.get("DEEP_PIT_M", "15"))
FLOOR_RAISE = __import__("os").environ.get("FLOOR_RAISE", "0") == "1"
MAIN_STEM_KM2 = float(__import__("os").environ.get("MAIN_STEM_KM2", "50"))
WIDEN_STEP_M = 0.5
WIDEN_PASSES = int(__import__("os").environ.get("WIDEN_PASSES", "1"))
WIDEN_TIERS = __import__("os").environ.get("WIDEN_TIERS", "")  # e.g. '50:3,3000:6,6000:12' (km2:passes)
GALCHHI_IN_DOMAIN_RUNOFF_M3_S = 450.0  # Trishuli at Galchhi ~600 minus Bhote Koshi boundary 150
CHANNEL_AREA_KM2 = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", type=float, default=60.0)
    p.add_argument("--margin", type=float, default=4000.0)
    p.add_argument("--tag", default="corridor60")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="UTM 45N sub-window; stations/inflows outside are skipped")
    p.add_argument("--runoff-mm-day", type=float, help="override the Galchhi-scaled uniform runoff")
    p.add_argument("--dem-override", help="GeoTIFF (EPSG:32645, EGM2008, same grid) whose valid cells replace the Copernicus DSM")
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority_flood_fill(z: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Barnes et al. (2014) priority-flood with epsilon; guarantees drainage to the edge."""
    rows, cols = z.shape
    filled = z.copy()
    visited = np.zeros_like(z, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in (0, cols - 1):
            heap.append((float(filled[r, c]), r, c)); visited[r, c] = True
    for c in range(1, cols - 1):
        for r in (0, rows - 1):
            heap.append((float(filled[r, c]), r, c)); visited[r, c] = True
    heapq.heapify(heap)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        zc, r, c = heapq.heappop(heap)
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and not visited[rr, cc]:
                visited[rr, cc] = True
                if filled[rr, cc] <= zc:
                    filled[rr, cc] = zc + eps
                heapq.heappush(heap, (float(filled[rr, cc]), rr, cc))
    return filled


def d8_receivers(filled: np.ndarray, cell: float) -> np.ndarray:
    """Index of the steepest-descent neighbour (flat index), -1 for edge outlets."""
    rows, cols = filled.shape
    big = np.full((rows + 2, cols + 2), -np.inf)
    big[1:-1, 1:-1] = filled
    best_drop = np.zeros_like(filled)
    best_rcv = np.full(filled.shape, -1, dtype=np.int64)
    idx = np.arange(rows * cols).reshape(rows, cols)
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        nz = big[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
        dist = cell * (np.sqrt(2.0) if dr and dc else 1.0)
        drop = (filled - nz) / dist
        rr = np.clip(np.arange(rows) + dr, 0, rows - 1)[:, None]
        cc = np.clip(np.arange(cols) + dc, 0, cols - 1)[None, :]
        cand = idx[rr, cc]
        inside = np.ones_like(filled, dtype=bool)
        if dr == -1: inside[0, :] = False
        if dr == 1: inside[-1, :] = False
        if dc == -1: inside[:, 0] = False
        if dc == 1: inside[:, -1] = False
        better = (drop > best_drop) & inside
        best_drop = np.where(better, drop, best_drop)
        best_rcv = np.where(better, cand, best_rcv)
    return best_rcv.ravel()


def accumulate(filled: np.ndarray, rcv: np.ndarray, channel_cells: int):
    """Total upslope cell count, plus hillslope area collected at channel cells."""
    order = np.argsort(filled.ravel())[::-1]
    acc = np.ones(filled.size, dtype=np.int64)
    rcv_list = rcv.tolist()
    acc_list = acc.tolist()
    for i in order.tolist():
        r = rcv_list[i]
        if r >= 0:
            acc_list[r] += acc_list[i]
    acc = np.asarray(acc_list, dtype=np.int64)
    channel = acc >= channel_cells
    lateral = np.ones(filled.size, dtype=np.int64)
    lat_list = lateral.tolist()
    ch_list = channel.tolist()
    for i in order.tolist():
        if ch_list[i]:
            continue
        r = rcv_list[i]
        if r >= 0:
            lat_list[r] += lat_list[i]
    lateral = np.asarray(lat_list, dtype=np.int64)
    lateral[~channel] = 0
    return acc.reshape(filled.shape), channel.reshape(filled.shape), lateral.reshape(filled.shape)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cell = args.cell
    to_utm = Transformer.from_crs(4326, 32645, always_xy=True)
    to_ll = Transformer.from_crs(32645, 4326, always_xy=True)

    route = gpd.read_file(ROUTE_PATH).to_crs(32645).geometry.iloc[0]
    source = gpd.read_file(SOURCE_PATH).to_crs(32645)
    center = shape(json.loads(CENTERLINE_PATH.read_text("utf-8"))["features"][0]["geometry"])
    bounds = np.array([g.bounds for g in (route, source.union_all(), center)])
    left, bottom = bounds[:, 0].min() - args.margin, bounds[:, 1].min() - args.margin
    right, top = bounds[:, 2].max() + args.margin, bounds[:, 3].max() + args.margin
    if args.bbox:
        left, right, bottom, top = args.bbox

    with rasterio.open(DEM_PATH) as dem:
        left, bottom = max(left, dem.bounds.left), max(bottom, dem.bounds.bottom)
        right, top = min(right, dem.bounds.right), min(top, dem.bounds.top)
        cols, rows = int(np.floor((right - left) / cell)), int(np.floor((top - bottom) / cell))
        right, bottom = left + cols * cell, top - rows * cell
        window = from_bounds(left, bottom, right, top, dem.transform)
        z = dem.read(1, window=window, out_shape=(rows, cols), masked=True,
                     resampling=Resampling.average).filled(np.nan).astype(np.float64)
    override_cells = 0
    if args.dem_override:
        with rasterio.open(args.dem_override) as ov:
            if (ov.width, ov.height) != (cols, rows):
                raise ValueError("dem override grid mismatch")
            o = ov.read(1).astype(np.float64); o[o == ov.nodata] = np.nan
        good = np.isfinite(o)
        z = np.where(good, o, z); override_cells = int(good.sum())
        print(f"dem override: {override_cells} cells ({good.mean()*100:.1f}%) from {args.dem_override}")
    nan_count = int(np.isnan(z).sum())
    if nan_count:
        z = np.where(np.isnan(z), np.nanmax(z) + 500.0, z)
    print(f"grid {rows}x{cols} cell {cell} m, nan cells {nan_count}")

    transform = rasterio.transform.from_origin(left, top, cell, cell)
    source_mask = rasterize([(g, 1) for g in source.geometry], out_shape=(rows, cols),
                            transform=transform, all_touched=False, dtype=np.uint8).astype(bool)
    source_area = source_mask.sum() * cell * cell
    thickness = SOURCE_VOLUME_M3 / source_area
    print(f"source cells {source_mask.sum()} area {source_area/1e6:.3f} km2 thickness {thickness:.2f} m")

    print("priority flood ...", flush=True)
    z_raw = z.copy()
    filled = priority_flood_fill(z)
    fill_depth = filled - z_raw
    print(f"pit fill: {int((fill_depth > 0.01).sum())} cells, volume {fill_depth.sum() * cell * cell / 1e6:.1f} Mm3, "
          f"max {fill_depth.max():.1f} m")
    rcv = d8_receivers(filled, cell)
    # Breach (carve) the raw DSM along D8 receivers so every drainage path descends by at least
    # MIN_SLOPE per cell: artificial dams are cut instead of pits being raised into flat lakes.
    # hybrid conditioning: deep depressions (> DEEP_PIT_M) are radar voids and are FILLED;
    # shallow ones (canopy/bridge dams) are breached by the carving below
    z_start = np.where(fill_depth > DEEP_PIT_M, filled, z_raw)
    print(f"deep-pit fill: {int((fill_depth > DEEP_PIT_M).sum())} cells, {fill_depth[fill_depth > DEEP_PIT_M].sum() * cell * cell / 1e6:.1f} Mm3")
    order = np.argsort(filled.ravel())[::-1]
    zc = z_start.ravel().copy()
    rcv_list = rcv.tolist(); zc_list = zc.tolist()
    channel_cells = int(CHANNEL_AREA_KM2 * 1e6 / (cell * cell))
    acc_pre, _, _ = accumulate(filled, rcv, channel_cells)
    is_channel = (acc_pre.ravel() >= channel_cells).tolist()
    pit_list = (fill_depth.ravel() > 0.5).tolist()
    diag_fixes = 0
    for i in order.tolist():
        r = rcv_list[i]
        if r < 0 or not is_channel[i]:
            continue
        ri, ci = divmod(i, cols)
        rr, cr = divmod(r, cols)
        if rr != ri and cr != ci:
            # diagonal step: the 4-connected flux scheme needs an orthogonal intermediate cell
            m1, m2 = rr * cols + ci, ri * cols + cr
            m = m1 if zc_list[m1] <= zc_list[m2] else m2
            floor_m = zc_list[i] - MIN_SLOPE * cell
            if zc_list[m] > floor_m:
                zc_list[m] = floor_m
                diag_fixes += 1
            elif FLOOR_RAISE and pit_list[m] and zc_list[m] < floor_m:
                zc_list[m] = floor_m
            floor = floor_m - MIN_SLOPE * cell
        else:
            floor = zc_list[i] - MIN_SLOPE * cell
        if zc_list[r] > floor:
            zc_list[r] = floor
        elif FLOOR_RAISE and pit_list[r] and zc_list[r] < floor:
            zc_list[r] = floor  # radar void below the descending profile: raise to the profile
    z = np.asarray(zc_list).reshape(rows, cols)
    print(f"diagonal connectivity fixes: {diag_fixes}")
    # main stem (>= MAIN_STEM_KM2) carved two cells wide: the lowest orthogonal neighbour is
    # brought down to the channel level so the conveying width is 2*cell (~120 m at 60 m)
    main_cells = int(MAIN_STEM_KM2 * 1e6 / (cell * cell))
    main = acc_pre >= main_cells
    channel_pre = acc_pre >= channel_cells
    zpad = np.pad(np.where(channel_pre, np.inf, z), 1, mode="edge")  # only non-channel neighbours are candidates
    nb = np.stack([zpad[:-2, 1:-1], zpad[2:, 1:-1], zpad[1:-1, :-2], zpad[1:-1, 2:]])
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    best = np.argmin(nb, axis=0)
    widened = 0
    grown = main.copy()
    tiers = [(float(a), int(n)) for a, n in (t.split(':') for t in WIDEN_TIERS.split(',') if t)] if WIDEN_TIERS else []
    max_passes = max([WIDEN_PASSES] + [n for _, n in tiers])
    for _pass in range(max_passes):
        # tiered width: cells of contributing area >= a km2 are widened by n passes
        if tiers:
            allowed = np.zeros_like(main)
            for a, n in tiers:
                if _pass < n:
                    allowed |= acc_pre >= int(a * 1e6 / (cell * cell))
            allowed = ndimage.binary_dilation(allowed, iterations=_pass + 1) if _pass else allowed
        else:
            allowed = np.ones_like(main)
        zpad = np.pad(np.where(grown | channel_pre, np.inf, z), 1, mode="edge")
        nb = np.stack([zpad[:-2, 1:-1], zpad[2:, 1:-1], zpad[1:-1, :-2], zpad[1:-1, 2:]])
        best = np.argmin(nb, axis=0)
        new_cells = np.zeros_like(grown)
        for r, c in zip(*np.nonzero(grown & allowed)):
            dr, dc = offs[best[r, c]]
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and np.isfinite(nb[best[r, c], r, c]) and z[rr, cc] > z[r, c] + WIDEN_STEP_M:
                z[rr, cc] = z[r, c] + WIDEN_STEP_M
                new_cells[rr, cc] = True
                widened += 1
        grown |= new_cells
    print(f"main-stem widening: {int(main.sum())} channel cells, {widened} neighbours lowered")
    carve = z_start - z
    if FLOOR_RAISE:
        # raise the interior of every breached depression to the lowest carved level inside it,
        # so that neither a lake (fill) nor a canyon (carve) can store the flood

        pit = fill_depth > 0.5
        lab, nlab = ndimage.label(pit)
        carved_cells = (carve > 0.01) | (acc_pre >= channel_cells)
        zc_masked = np.where(carved_cells & pit, z, np.inf)
        floor = ndimage.minimum(zc_masked, lab, index=np.arange(1, nlab + 1))
        floor_map = np.full(z.shape, -np.inf); has = np.isfinite(floor)
        idx = np.arange(1, nlab + 1)[has]
        for k, f in zip(idx, floor[has]):
            pass
        floor_map = np.where(lab > 0, np.take(np.r_[-np.inf, np.where(np.isfinite(floor), floor, -np.inf)], lab), -np.inf)
        raise_to = np.maximum(z, floor_map)
        raise_to = np.where(carved_cells, z, raise_to)
        raised = raise_to - z
        z = raise_to
        print(f"floor-raise: {int((raised > 0.01).sum())} cells, {raised.sum() * cell * cell / 1e6:.1f} Mm3")
    print(f"carve: {int((carve > 0.01).sum())} cells, volume {carve.sum() * cell * cell / 1e6:.1f} Mm3, max {carve.max():.1f} m")
    channel_cells = int(CHANNEL_AREA_KM2 * 1e6 / (cell * cell))
    acc, channel, lateral = accumulate(filled, rcv, channel_cells)
    print(f"channel cells {channel.sum()}, max accumulation {acc.max()*cell*cell/1e6:.0f} km2")

    def rc_of(x: float, y: float) -> tuple[int, int]:
        return int((top - y) // cell), int((x - left) // cell)

    def snap_to_channel(x: float, y: float, radius_m: float) -> tuple[int, int]:
        r0, c0 = rc_of(x, y)
        k = int(radius_m // cell)
        rs, cs = slice(max(r0 - k, 0), r0 + k + 1), slice(max(c0 - k, 0), c0 + k + 1)
        sub = acc[rs, cs]
        rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
        return rs.start + rr, cs.start + cc

    def inside(x: float, y: float) -> bool:
        return left + 2 * cell < x < right - 2 * cell and bottom + 2 * cell < y < top - 2 * cell

    stations = {}
    obs = pd.read_csv(OBS_PATH).set_index("id")
    for oid, label in [("usgs_main_onset", "Seismic origin"), ("gyirong_cctv_arrival", "Gyirong CCTV"),
                       ("rasuwagadhi_signal_loss", "Rasuwagadhi"), ("syabrubesi_signal_loss", "Syabrubesi")]:
        row = obs.loc[oid]
        x, y = to_utm.transform(row.longitude, row.latitude)
        if not inside(x, y):
            continue
        r, c = (rc_of(x, y) if oid == "usgs_main_onset" else snap_to_channel(x, y, 300.0))
        stations[oid] = {"label": label, "lon": float(row.longitude), "lat": float(row.latitude),
                         "x": x, "y": y, "row": int(r), "col": int(c),
                         "start_utc": row.start_utc, "end_utc": row.end_utc}
    offsets = pd.read_csv(OFFSETS_PATH)
    offsets = offsets[offsets.route_variant == "mapped_candidate"].set_index("id")
    for oid, label in [("galchhi_gauge", "Galchhi"), ("kali_khola_gauge", "Kali Khola"),
                       ("devghat_2023_candidate", "Devghat (candidate)")]:
        row = offsets.loc[oid]
        x, y = to_utm.transform(row.lon, row.lat)
        if not inside(x, y):
            continue
        r, c = snap_to_channel(x, y, 400.0)
        stations[oid] = {"label": label, "lon": float(row.lon), "lat": float(row.lat), "x": x, "y": y,
                         "row": int(r), "col": int(c), "chainage_m": float(row.chainage_m)}

    # External boundary inflows: lowest edge cell within a lon/lat window.
    inflows = []
    outlet_key = "devghat_2023_candidate" if "devghat_2023_candidate" in stations else list(stations)[-1]
    for name, spec in EXTERNAL_INFLOWS.items():
        if "anchor_lonlat" in spec:
            ax_, ay_ = to_utm.transform(*spec["anchor_lonlat"])
            if not inside(ax_, ay_):
                print(f"  {name}: anchor outside window, skipped")
                continue
            r, c = snap_to_channel(ax_, ay_, spec["snap_m"])
        elif spec["edge"] == "north":
            xs = left + (np.arange(cols) + 0.5) * cell
            lons = np.array([to_ll.transform(x, top - cell / 2)[0] for x in xs])
            sel = np.where((lons >= spec["lon_range"][0]) & (lons <= spec["lon_range"][1]))[0]
            if not len(sel):
                print(f"  {name}: edge window outside domain, skipped")
                continue
            c = int(sel[np.argmin(z[0, sel])]); r = 0
        else:
            ys = top - (np.arange(rows) + 0.5) * cell
            lats = np.array([to_ll.transform(left + cell / 2, y)[1] for y in ys])
            sel = np.where((lats >= spec["lat_range"][0]) & (lats <= spec["lat_range"][1]))[0]
            r = int(sel[np.argmin(z[sel, 0])]); c = 0
        # inject on the main channel a few rows inside the domain and verify it drains to Devghat
        dv = stations[outlet_key]
        target = dv["row"] * cols + dv["col"]
        r2, c2 = r, c
        if "anchor_lonlat" in spec:
            pass
        elif spec["edge"] == "north":
            sub = acc[INFLOW_INSET_CELLS, max(c - 25, 0):c + 26]
            c2 = max(c - 25, 0) + int(np.argmax(sub)); r2 = INFLOW_INSET_CELLS
        else:
            sub = acc[max(r - 25, 0):r + 26, INFLOW_INSET_CELLS]
            r2 = max(r - 25, 0) + int(np.argmax(sub)); c2 = INFLOW_INSET_CELLS
        node = r2 * cols + c2
        reaches = False
        for _ in range(rows * cols):
            if node == target:
                reaches = True
                break
            node = rcv[node]
            if node < 0:
                break
        spec["reaches_devghat"] = reaches
        print(f"  {name}: D8 path from injection reaches {outlet_key} cell: {reaches}")
        lon, lat = to_ll.transform(left + (c2 + 0.5) * cell, top - (r2 + 0.5) * cell)
        inflows.append({"name": name, "row": int(r2), "col": int(c2), "q_m3_s": spec["q_m3_s"],
                        "lon": float(lon), "lat": float(lat), "edge_elevation_m": float(z[r, c]),
                        "d8_path_reaches_devghat": reaches})
        print(f"inflow {name}: r={r2} c={c2} z={z[r,c]:.0f} lon={lon:.4f} lat={lat:.4f}")

    if args.runoff_mm_day is not None:
        galchhi_area_m2 = float("nan")
        runoff_m_s = args.runoff_mm_day / 1000.0 / 86400.0
    else:
        g = stations["galchhi_gauge"]
        galchhi_area_m2 = float(acc[g["row"], g["col"]]) * cell * cell
        runoff_m_s = GALCHHI_IN_DOMAIN_RUNOFF_M3_S / galchhi_area_m2
    lateral_q = lateral.astype(np.float64) * cell * cell * runoff_m_s
    print(f"in-domain Galchhi catchment {galchhi_area_m2/1e6:.0f} km2, runoff {runoff_m_s*86400e3:.2f} mm/day, "
          f"lateral total {lateral_q.sum():.0f} m3/s")

    # route chainage per cell (within 400 m of mapped route or observed centreline)
    from scipy.spatial import cKDTree
    xs = left + (np.arange(cols) + 0.5) * cell
    ys = top - (np.arange(rows) + 0.5) * cell
    xx, yy = np.meshgrid(xs, ys)

    def chainage_field(line: LineString, radius: float) -> np.ndarray:
        d = np.arange(0.0, line.length + 30.0, 30.0)
        pts = np.array([line.interpolate(s).coords[0] for s in d])
        tree = cKDTree(pts)
        dist, ind = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
        ch = np.where(dist <= radius, d[ind], np.nan)
        return ch.reshape(rows, cols)

    route_chainage = chainage_field(route, 400.0)
    upper_chainage = chainage_field(center, 400.0)

    np.savez_compressed(OUT / f"{args.tag}.npz", z=z.astype(np.float32), z_raw=z_raw.astype(np.float32),
                        fill_depth=fill_depth.astype(np.float32), carve_depth=carve.astype(np.float32),
                        source_mask=source_mask,
                        source_thickness_m=np.float32(thickness), channel=channel,
                        lateral_q_m3_s=lateral_q.astype(np.float32), acc_cells=acc.astype(np.int32),
                        rcv=rcv.astype(np.int32),
                        route_chainage_m=route_chainage.astype(np.float32),
                        upper_chainage_m=upper_chainage.astype(np.float32))
    meta = {
        "tag": args.tag, "cell_m": cell, "rows": rows, "cols": cols,
        "extent_utm45n": [left, right, bottom, top], "crs": "EPSG:32645", "vertical_datum": "EGM2008",
        "dem_source": str(DEM_PATH), "dem_sha256": sha256(DEM_PATH), "dem_nan_cells_replaced": nan_count,
        "dem_override": args.dem_override, "dem_override_cells": override_cells,
        "simulation_bed": f"raw DSM breached along priority-flood D8 receivers with minimum slope {MIN_SLOPE}; raw DSM kept as z_raw",
        "pit_fill_cells_diagnostic": int((fill_depth > 0.01).sum()), "pit_fill_volume_m3_diagnostic": float(fill_depth.sum() * cell * cell),
        "pit_fill_max_m_diagnostic": float(fill_depth.max()),
        "carve_cells": int((carve > 0.01).sum()), "carve_volume_m3": float(carve.sum() * cell * cell),
        "carve_max_m": float(carve.max()), "min_slope": MIN_SLOPE,
        "source_polygon": str(SOURCE_PATH), "source_sha256": sha256(SOURCE_PATH),
        "source_volume_m3": SOURCE_VOLUME_M3, "source_cells": int(source_mask.sum()),
        "source_area_m2": source_area, "source_thickness_m": thickness,
        "source_elevation_range_m": [float(z[source_mask].min()), float(z[source_mask].max())],
        "external_inflows": inflows, "galchhi_in_domain_catchment_m2": galchhi_area_m2,
        "uniform_runoff_m_s": runoff_m_s, "channel_threshold_km2": CHANNEL_AREA_KM2,
        "stations": stations, "route_length_m": route.length, "onset_utc": "2026-08-26T02:52:10Z",
        "assumptions": [
            "Single-layer mixture shallow-water model; no phase separation, melt or entrainment.",
            "Source thickness uniform over the UNOSAT detachment zone at the QHNU 35.42 Mm3 volume.",
            "Baseflow: uniform in-domain runoff scaled to 450 m3/s at Galchhi plus four boundary inflows;",
            "boundary discharges are conditional monsoon assumptions, not measurements.",
        ],
    }
    (OUT / f"{args.tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT / f"{args.tag}.npz")


if __name__ == "__main__":
    main()
