"""Active-cell mask for the sparse solver: corridor band around the main channel network plus source slopes.

Lateral runoff of channel cells outside the band is routed along D8 receivers to the first active cell.
"""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--band-m", type=float, default=600.0, help="half-width of the band around route/centreline")
    p.add_argument("--network-km2", type=float, default=20.0, help="tributaries with this contributing area get a narrower band")
    p.add_argument("--network-band-m", type=float, default=240.0)
    p.add_argument("--source-band-m", type=float, default=900.0)
    a = p.parse_args()
    d = dict(np.load(ROOT / "inputs" / f"{a.tag}.npz"))
    meta = json.loads((ROOT / "inputs" / f"{a.tag}.json").read_text("utf-8"))
    cell = meta["cell_m"]; rows, cols = d["z"].shape
    corridor = np.isfinite(d["route_chainage_m"]) | np.isfinite(d["upper_chainage_m"])
    acc_km2 = d["acc_cells"].astype(np.float64) * cell * cell / 1e6
    network = acc_km2 >= a.network_km2
    k1 = int(round(a.band_m / cell)); k2 = int(round(a.network_band_m / cell)); k3 = int(round(a.source_band_m / cell))
    active = ndimage.binary_dilation(corridor, iterations=k1) | ndimage.binary_dilation(network, iterations=k2) \
        | ndimage.binary_dilation(d["source_mask"], iterations=k3)
    active |= ndimage.binary_dilation(d["channel"], iterations=1)  # whole channel network, 4-connected
    # keep the connected component(s) that touch the corridor
    lab, n = ndimage.label(active)
    keep = np.unique(lab[corridor | d["source_mask"]]); keep = keep[keep > 0]
    active = np.isin(lab, keep)
    # route runoff of inactive channel cells to the first active cell downstream
    lat = d["lateral_q_m3_s"].astype(np.float64).ravel(); rcv = d["rcv"].astype(np.int64).ravel(); act = active.ravel()
    lat_active = np.where(act, lat, 0.0)
    lost = 0.0; moved = 0.0
    for i in np.nonzero((lat > 0) & ~act)[0]:
        j = int(i); steps = 0
        while j >= 0 and not act[j] and steps < 100000:
            j = int(rcv[j]); steps += 1
        if j >= 0 and act[j]:
            lat_active[j] += lat[i]; moved += lat[i]
        else:
            lost += lat[i]
    # external inflow points must be active: move each inactive one down its D8 path to the first active cell
    for inflow in meta["external_inflows"]:
        j = inflow["row"] * cols + inflow["col"]; steps = 0
        while j >= 0 and not act[j] and steps < 100000:
            j = int(rcv[j]); steps += 1
        if j >= 0 and act[j] and j != inflow["row"] * cols + inflow["col"]:
            inflow["row"], inflow["col"] = int(j // cols), int(j % cols); inflow["moved_to_active"] = True
            print(f"  inflow {inflow['name']} moved to active cell {inflow['row']},{inflow['col']}")
    d["active"] = active; d["lateral_q_active_m3_s"] = lat_active.reshape(rows, cols).astype(np.float32)
    np.savez_compressed(ROOT / "inputs" / f"{a.tag}.npz", **d)
    meta["active_cells"] = int(active.sum()); meta["active_fraction"] = float(active.mean())
    meta["active_band_m"] = a.band_m; meta["runoff_rerouted_m3_s"] = moved; meta["runoff_lost_outside_m3_s"] = lost
    (ROOT / "inputs" / f"{a.tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{a.tag}: active {active.sum()} of {active.size} cells ({active.mean()*100:.1f}%), runoff rerouted {moved:.0f} m3/s, lost {lost:.0f} m3/s")


if __name__ == "__main__":
    main()
