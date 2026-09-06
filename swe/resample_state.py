"""Nearest-neighbour resampling of a 60 m spin-up state onto the 30 m grid as a restart file."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-run", required=True); p.add_argument("--src-tag", default="corridor60h")
    p.add_argument("--dst-tag", default="corridor30h"); p.add_argument("--out", required=True)
    a = p.parse_args()
    ms = json.loads((ROOT / "inputs" / f"{a.src_tag}.json").read_text("utf-8")); md = json.loads((ROOT / "inputs" / f"{a.dst_tag}.json").read_text("utf-8"))
    st = np.load(ROOT / "runs" / a.src_run / "final_state.npz")
    ls, rs, bs, ts = ms["extent_utm45n"]; ld, rd, bd, td = md["extent_utm45n"]
    cs, cd = ms["cell_m"], md["cell_m"]
    rows_d, cols_d = md["rows"], md["cols"]
    xd = ld + (np.arange(cols_d) + 0.5) * cd; yd = td - (np.arange(rows_d) + 0.5) * cd
    cj = np.clip(((xd - ls) // cs).astype(int), 0, ms["cols"] - 1); ri = np.clip(((ts - yd) // cs).astype(int), 0, ms["rows"] - 1)
    out = {}
    for k in ("h", "hu", "hv", "hc", "hi"):
        if k in st:
            out[k] = st[k][ri][:, cj].astype(np.float32)
    # keep only water on channel-ish cells of the fine grid: cells whose fine bed is within 3 m of the coarse water surface
    zd = np.load(ROOT / "inputs" / f"{a.dst_tag}.npz")["z"]
    zs = np.load(ROOT / "inputs" / f"{a.src_tag}.npz")["z"][ri][:, cj]
    surf = zs + out["h"]
    h_new = np.clip(surf - zd, 0, None).astype(np.float32)
    h_new[out["h"] <= 0] = 0
    scale = np.where(out["h"] > 0, h_new / np.maximum(out["h"], 1e-3), 0).astype(np.float32)
    out["h"] = h_new
    for k in ("hu", "hv", "hc", "hi"):
        if k in out:
            out[k] = out[k] * scale
    np.savez_compressed(a.out, **out, time_s=0.0)
    print("wrote", a.out, "volume", out["h"].sum() * cd * cd / 1e6, "Mm3")


if __name__ == "__main__":
    main()
