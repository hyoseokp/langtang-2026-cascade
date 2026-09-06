"""Recompute station transect series (Q, stage, hmax, cmax) from stored frames with current transects."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--run-id", required=True); p.add_argument("--tag", default="corridor60")
    p.add_argument("--base-run")
    a = p.parse_args()
    inp = np.load(ROOT / "inputs" / f"{a.tag}.npz"); z = inp["z"].astype(np.float32); n = z.size; cell = 60.0 if "60" in a.tag else 30.0
    tr = json.loads((ROOT / "inputs" / f"{a.tag}_transects.json").read_text("utf-8"))
    for t in tr.values():
        t["idx"] = np.array(t["rows"]) * z.shape[1] + np.array(t["cols"])
    series = pd.read_csv(ROOT / "runs" / a.run_id / "series.csv")
    frames = sorted((ROOT / "runs" / a.run_id / "frames").glob("frame_*.npz"))
    rows = []
    zf = z.ravel()
    for fp in frames:
        fr = np.load(fp)
        h = np.zeros(n, np.float32); hu = np.zeros(n, np.float32); hv = np.zeros(n, np.float32); c = np.zeros(n, np.float32)
        idx = fr["idx"]; h[idx] = fr["h"]; hu[idx] = fr["hu"]; hv[idx] = fr["hv"]; c[idx] = fr["c"]
        row = {"time_s": float(fr["time_s"])}
        for key, t in tr.items():
            i = t["idx"]; tx, ty = t["tangent"]
            row[f"{key}_Q"] = float(((hu[i] * tx + hv[i] * ty) * cell * t.get("flux_scale", 1.0)).sum())
            row[f"{key}_Qdebris"] = float(((hu[i] * tx + hv[i] * ty) * cell * t.get("flux_scale", 1.0) * c[i]).sum())
            ht = h[i]; wet = ht > 0.05
            row[f"{key}_hmax"] = float(ht.max()); row[f"{key}_cmax"] = float(c[i][wet].max()) if wet.any() else 0.0
            row[f"{key}_stage"] = float((zf[i] + ht)[wet].min()) if wet.any() else np.nan
            row[f"{key}_wet_width_m"] = float(wet.sum() * cell)
        rows.append(row)
    new = pd.DataFrame(rows)
    keep = [c for c in series.columns if not any(c.startswith(k + "_") for k in tr)]
    base_df = series[keep].copy(); base_df["time_s"] = base_df["time_s"].round(0); new["time_s"] = new["time_s"].round(0)
    merged = base_df.merge(new, on="time_s", how="inner")
    out = ROOT / "runs" / a.run_id / "series_transects_v2.csv"
    merged.to_csv(out, index=False)
    print("wrote", out, len(merged), "rows")
    if a.base_run:
        base = ROOT / "runs" / a.base_run / "final_state.npz"
        st = np.load(base); h = st["h"].ravel(); hu = st["hu"].ravel(); hv = st["hv"].ravel()
        brow = {}
        for key, t in tr.items():
            i = t["idx"]; tx, ty = t["tangent"]
            brow[f"{key}_Q"] = float(((hu[i] * tx + hv[i] * ty) * cell * t.get("flux_scale", 1.0)).sum())
            ht = h[i]; wet = ht > 0.05
            brow[f"{key}_hmax"] = float(ht.max()); brow[f"{key}_stage"] = float((zf[i] + ht)[wet].min()) if wet.any() else np.nan
        pd.DataFrame([brow]).to_csv(ROOT / "runs" / a.base_run / "baseline_transects_v2.csv", index=False)
        print("baseline:", {k: round(v, 1) for k, v in brow.items() if k.endswith("_Q")})


if __name__ == "__main__":
    main()
