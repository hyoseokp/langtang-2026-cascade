"""Independent valley-floor width below Syabrubesi from the raw DSM: cells within dz of the pre-event water surface,
within 1.5 km of the route, per 2-km route segment; compared with the hydraulic-geometry channel width."""

from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "swe"))
from synth_channel import width_from_area  # noqa: E402


def main() -> None:
    d = np.load(ROOT / "inputs" / "corridor60s.npz"); meta = json.loads((ROOT / "inputs" / "corridor60s.json").read_text("utf-8")); cell = meta["cell_m"]
    zr = d["z_raw"].astype(np.float64); z = d["z"].astype(np.float64); rc = d["route_chainage_m"]; acc = d["acc_cells"].astype(np.float64) * cell * cell / 1e6
    hb = np.load(ROOT / "runs" / "spinup60s_c" / "final_state.npz")["h"].astype(np.float64)
    chan = d["channel"] & (acc > 500)
    wsurf = np.where(chan & (hb > 0.2), z + hb, np.nan)
    # water level field: nearest channel cell's surface
    dist, (ri, ci) = ndimage.distance_transform_edt(~np.isfinite(wsurf), return_indices=True)
    wl = wsurf[ri, ci]; near = dist * cell <= 1500
    rows = []
    sy = 13800.0  # Syabrubesi route chainage (m)
    for s0 in np.arange(sy, np.nanmax(rc), 2000.0):
        seg = np.isfinite(rc) & (rc >= s0) & (rc < s0 + 2000) & chan
        if not seg.any():
            continue
        # cells attributed to this segment: nearest channel cell lies in the segment
        attrib = near & seg[ri, ci]
        rec = {"route_km_from_syabrubesi": (s0 - sy) / 1e3, "channel_width_m": float(width_from_area(acc[seg]).mean())}
        for dz in (2.0, 4.0, 8.0):
            floor = attrib & (zr <= wl + dz)
            rec[f"floor_width_dz{dz:.0f}_m"] = float(floor.sum()) * cell * cell / 2000.0
        rows.append(rec)
    df = pd.DataFrame(rows); out = ROOT / "figures" / "paper" / "table_valley_floor_width.csv"; df.to_csv(out, index=False)
    for lo, hi, name in ((0, 32, "Syabrubesi-Betrawati"), (32, 66, "Betrawati-Galchhi"), (66, 135, "Galchhi-Kali Khola"), (135, 170, "Kali Khola-Devghat")):
        sub = df[(df.route_km_from_syabrubesi >= lo) & (df.route_km_from_syabrubesi < hi)]
        print(f"{name:22s} channel W {sub.channel_width_m.median():5.0f} m | floor within +2 m {sub.floor_width_dz2_m.median():5.0f} m, +4 m {sub.floor_width_dz4_m.median():5.0f} m, +8 m {sub.floor_width_dz8_m.median():5.0f} m "
              f"| ratio +4 m / W = {(sub.floor_width_dz4_m / sub.channel_width_m).median():.2f}")


if __name__ == "__main__":
    main()
