"""Model maximum flow height along the corridor vs satellite trimline heights (stored frames only)."""

from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "swe"))
from render_profile import build_profile, sample  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="upper30"); p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--base-run", default="u30_spinup_v1")
    a = p.parse_args()
    inp = np.load(ROOT / "inputs" / f"{a.tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{a.tag}.json").read_text("utf-8"))
    z = inp["z"].astype(np.float32)
    chain, bed, nb, stations, join_km = build_profile(meta, z, inp=inp)
    hb = np.load(ROOT / "runs" / a.base_run / "final_state.npz")["h"].astype(np.float32).ravel()[nb].max(axis=1)
    trim = pd.read_csv(ROOT / "inputs" / f"{a.tag}_trimline_profile.csv")
    # profile chainage -> (line, line chainage): upper centreline before the join, route after it
    obs = []
    for _, r in trim.iterrows():
        km = r.chainage_km if r.line == "upper" else r.chainage_km + join_km
        if (r.line == "upper" and r.chainage_km > join_km) or (r.line == "route" and False):
            continue
        obs.append({"km": round(km * 2) / 2, "planet": r.planet_disturbed_p50_trimline_m, "unosat": r.unosat_affected_trimline_m,
                    "planet_width": r.planet_disturbed_p50_width_m})
    obs = pd.DataFrame(obs).groupby("km").first().reset_index()
    summary = []
    for run in a.runs:
        frames = sorted((ROOT / "runs" / run / "frames").glob("frame_*.npz"))
        if not frames:
            print(run, "no frames"); continue
        hmax = np.zeros(len(chain), np.float32); hend = None
        for fp in frames:
            h, c = sample(np.load(fp), nb, z.size)[:2]; hmax = np.maximum(hmax, h); hend = h
        rows = []
        for lo in np.arange(0, chain.max(), 0.5):
            m = (chain >= lo) & (chain < lo + 0.5)
            if m.any():
                rows.append({"km": lo, "model_hmax": float(hmax[m].max()), "model_hend": float(hend[m].max()), "base": float(hb[m].max())})
        d = pd.DataFrame(rows).merge(obs, on="km", how="left")
        d.to_csv(ROOT / "runs" / run / "max_height_profile.csv", index=False)
        pl = d.dropna(subset=["planet"]); pl = pl[pl.planet > 3]
        s = {"run": run, "bins_total": len(d), "model_hmax_median_0_22km": float(d[d.km < join_km].model_hmax.median()),
             "model_hmax_median_route": float(d[d.km >= join_km].model_hmax.median()),
             "model_hend_median": float(d.model_hend.median()), "planet_bins": len(pl)}
        if len(pl):
            s.update({"planet_trimline_median": float(pl.planet.median()), "model_at_planet_median": float(pl.model_hmax.median()),
                      "ratio_median": float((pl.model_hmax / pl.planet).median()),
                      "corr": float(np.corrcoef(pl.model_hmax, pl.planet)[0, 1]) if len(pl) > 2 else np.nan})
        summary.append(s)
    df = pd.DataFrame(summary); pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
    df.to_csv(ROOT / "figures" / "paper" / "table_height_vs_trimline.csv", index=False)


if __name__ == "__main__":
    main()
