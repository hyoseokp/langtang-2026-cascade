"""Compare modelled bed change (entrainment) with Geo-PERA measured elevation change (2 m) on the model grid."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
GEO = ROOT / "data/data_raw/geo_pera/v1.1"
AREAS = {"rasuwagadhi_timure": "bhotekoshi_dh_rasuwagadhi-timure_2m.tif", "syabrubesi": "bhotekoshi_dh_syabrubesi_2m.tif"}


def observed_on_grid(tag: str):
    meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    cell = meta["cell_m"]; left, right, bottom, top = meta["extent_utm45n"]
    shape = (meta["rows"], meta["cols"])
    transform = rasterio.transform.from_origin(left, top, cell, cell)
    out = {}
    for name, fn in AREAS.items():
        with rasterio.open(GEO / fn) as src:
            dh = np.full(shape, np.nan, np.float32)
            reproject(src.read(1), dh, src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                      dst_crs="EPSG:32645", resampling=Resampling.average, src_nodata=src.nodata, dst_nodata=np.nan)
            valid = np.zeros(shape, np.float32)
            reproject((src.read(1, masked=True).mask == False).astype(np.float32), valid, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=transform, dst_crs="EPSG:32645", resampling=Resampling.average)
        out[name] = {"dh": dh, "valid": valid > 0.5}
    path = ROOT / "inputs" / f"{tag}_geopera_dh.npz"
    np.savez_compressed(path, **{f"{k}_dh": v["dh"] for k, v in out.items()}, **{f"{k}_valid": v["valid"] for k, v in out.items()})
    return out, cell


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="upper30"); p.add_argument("--runs", nargs="*", default=[])
    a = p.parse_args()
    obs, cell = observed_on_grid(a.tag)
    area = cell * cell
    print(f"observed (Geo-PERA v1.1 on {cell:g} m grid):")
    for name, o in obs.items():
        v = o["dh"][o["valid"]]
        print(f"  {name}: net {np.nansum(v)*area/1e6:8.2f} Mm3 | deposit(dh>0.5) {np.nansum(v[v>0.5])*area/1e6:6.2f} Mm3 mean {np.nanmean(v[v>0.5]):.2f} m | "
              f"erosion(dh<-0.5) {np.nansum(v[v<-0.5])*area/1e6:7.2f} Mm3 mean {np.nanmean(v[v<-0.5]):.2f} m | cells {o['valid'].sum()}")
    for run in a.runs:
        st = np.load(ROOT / "runs" / run / "final_state.npz")
        if "bed_change" not in st:
            print(run, "has no bed_change"); continue
        bc = st["bed_change"].astype(np.float32); h = st["h"]; hc = st["hc"]
        deposit_proxy = bc + np.where(h > 0.05, hc, 0.0)  # remaining mobile debris counted as potential deposit
        print(f"model {run}:")
        for name, o in obs.items():
            m = o["valid"]; v = o["dh"][m]; b = bc[m]; d = deposit_proxy[m]
            corr = np.corrcoef(np.nan_to_num(v), b)[0, 1] if b.std() > 0 else np.nan
            print(f"  {name}: net bed change {b.sum()*area/1e6:8.2f} Mm3 (obs {np.nansum(v)*area/1e6:.2f}) | "
                  f"eroded {b[b<0].sum()*area/1e6:7.2f} | resting debris {np.where(h[m]>0.05, hc[m], 0).sum()*area/1e6:.2f} Mm3 | "
                  f"max resting debris depth {np.where(h[m]>0.05, hc[m], 0).max():.1f} m | cellwise corr(dh, bed change) {corr:.2f}")


if __name__ == "__main__":
    main()
