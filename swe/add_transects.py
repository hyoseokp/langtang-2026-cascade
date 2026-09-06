"""Define valley-crossing transects at each station for discharge and stage extraction."""

from __future__ import annotations

import os

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, shape

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
UP = ROOT / "data"
HALF_WIDTH_M = 900.0


def main(tag: str = "corridor60") -> None:
    meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    inp = np.load(ROOT / "inputs" / f"{tag}.npz")
    z = inp["z"]
    cell = meta["cell_m"]
    left, right, bottom, top = meta["extent_utm45n"]
    rows, cols = z.shape
    route = gpd.read_file(UP / "data_processed/terrain/downstream_route_osm_candidate.geojson").to_crs(32645).geometry.iloc[0]
    center = shape(json.loads((UP / "data_processed/terrain/observed_corridor_centerline.geojson").read_text("utf-8"))["features"][0]["geometry"])
    from pyproj import Transformer
    to_utm = Transformer.from_crs(4326, 32645, always_xy=True)
    extras = {"betrawati_gauge": (85.18, 27.97, "Betrawati"), "malekhu_gauge": (84.8272, 27.8137, "Malekhu")}
    stations = dict(meta["stations"])
    for key, (lon, lat, label) in extras.items():
        x, y = to_utm.transform(lon, lat)
        if left < x < right and bottom < y < top:
            stations[key] = {"label": label, "x": x, "y": y, "row": int((top - y) // cell), "col": int((x - left) // cell)}
    transects = {}
    for key, s in stations.items():
        if key == "usgs_main_onset":
            continue
        pt = Point(s["x"], s["y"])
        line = route if route.distance(pt) < center.distance(pt) or key not in ("gyirong_cctv_arrival",) else center
        # use whichever line passes closer; the CCTV proxy lies on the Bhote Koshi mapped route
        line = route if route.distance(pt) <= center.distance(pt) else center
        d = line.project(pt)
        p0 = np.array(line.interpolate(max(d - 150, 0)).coords[0])
        p1 = np.array(line.interpolate(min(d + 150, line.length)).coords[0])
        tangent = p1 - p0
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        base = np.array(line.interpolate(d).coords[0])
        n = int(2 * HALF_WIDTH_M / cell) + 1
        offsets = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, n)
        pts = base[None, :] + offsets[:, None] * normal[None, :]
        rr = ((top - pts[:, 1]) // cell).astype(int)
        cc = ((pts[:, 0] - left) // cell).astype(int)
        ok = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        rr, cc = rr[ok], cc[ok]
        # 4-connected staircase: insert an orthogonal intermediate cell at every diagonal step so
        # that a 4-connected flow cannot cross the transect between sampled cells
        cells = []
        for i in range(len(rr)):
            if i and rr[i] != rr[i - 1] and cc[i] != cc[i - 1]:
                cells.append((int(rr[i]), int(cc[i - 1])))
            cells.append((int(rr[i]), int(cc[i])))
        uniq = list(dict.fromkeys(cells))
        rr = np.array([u[0] for u in uniq]); cc = np.array([u[1] for u in uniq])
        flux_scale = 1.0 / (abs(float(normal[0])) + abs(float(normal[1])))
        zt = z[rr, cc]
        transects[key] = {
            "line": "route" if line is route else "centerline", "chainage_m": float(d),
            "tangent": tangent.tolist(), "rows": rr.tolist(), "cols": cc.tolist(), "flux_scale": flux_scale,
            "thalweg_row": int(rr[np.argmin(zt)]), "thalweg_col": int(cc[np.argmin(zt)]),
            "thalweg_z": float(zt.min()), "station_z": float(z[s["row"], s["col"]]),
            "distance_station_to_line_m": float(line.distance(pt)),
        }
        print(f"{key}: {transects[key]['line']} chainage {d/1e3:.2f} km, {len(rr)} cells, thalweg z {zt.min():.0f}, "
              f"station cell z {transects[key]['station_z']:.0f}, offset {line.distance(pt):.0f} m")
    (ROOT / "inputs" / f"{tag}_transects.json").write_text(json.dumps(transects), encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "corridor60")
