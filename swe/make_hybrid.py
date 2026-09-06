"""Assemble the hybrid run: 2-D corridor series for the gorge stations, 1-D routed series for the stations below Syabrubesi."""

from __future__ import annotations

import os

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
DOWN = ["betrawati_gauge", "galchhi_gauge", "malekhu_gauge", "kali_khola_gauge", "devghat_2023_candidate"]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--two-d", default="prod60"); p.add_argument("--one-d", required=True); p.add_argument("--out", default="hybrid")
    a = p.parse_args()
    s2 = pd.read_csv(ROOT / "runs" / a.two_d / "series.csv"); s1 = pd.read_csv(ROOT / "runs" / a.one_d / "series.csv")
    s1 = s1.set_index("time_s").reindex(s2.time_s).interpolate().reset_index()
    for st in DOWN:
        for q in ("Q", "Qdebris", "stage", "hmax", "cmax", "wet_width_m"):
            s2[f"{st}_{q}"] = s1[f"{st}_{q}"].to_numpy()
    out = ROOT / "runs" / a.out; out.mkdir(exist_ok=True)
    s2.to_csv(out / "series.csv", index=False)
    r2 = json.loads((ROOT / "runs" / a.two_d / "result.json").read_text("utf-8")); r1 = json.loads((ROOT / "runs" / a.one_d / "result.json").read_text("utf-8"))
    r2["hybrid"] = {"two_d": a.two_d, "one_d": a.one_d, "one_d_args": r1["args"], "one_d_deposited_m3": r1["deposited_m3"]}
    (out / "result.json").write_text(json.dumps(r2, indent=1), encoding="utf-8")
    for name in ("final_state.npz", "max_footprint.npz"):
        if (ROOT / "runs" / a.two_d / name).exists():
            shutil.copy(ROOT / "runs" / a.two_d / name, out / name)
    print("wrote", out)


if __name__ == "__main__":
    main()
