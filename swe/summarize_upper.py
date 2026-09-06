"""Tabulate upper-corridor arrival times for a set of runs (stored series only)."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
SITES = [("gyirong_cctv_arrival", "G"), ("rasuwagadhi_signal_loss", "R"), ("syabrubesi_signal_loss", "S")]
OBS = {"G": (373, 553), "R": (170, 770), "S": (770, 1370)}


def first(t: np.ndarray, cond: np.ndarray) -> float:
    i = np.nonzero(cond)[0]
    return float(t[i[0]]) if len(i) else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--base-run", help="spin-up run providing the pre-event stage/discharge")
    p.add_argument("--out", default="figures/paper/table_upper_arrivals.csv")
    a = p.parse_args()
    base = pd.read_csv(ROOT / "runs" / a.base_run / "series.csv").iloc[-1] if a.base_run else None
    rows = []
    for r in a.runs:
        path = ROOT / "runs" / r / "series.csv"
        if not path.exists():
            print(r, "missing"); continue
        s = pd.read_csv(path); t = s.time_s.to_numpy()
        row = {"run": r}
        for st, k in SITES:
            stage0 = float(base[f"{st}_stage"]) if base is not None and np.isfinite(base[f"{st}_stage"]) else np.nan
            q0 = abs(float(base[f"{st}_Q"])) if base is not None else 0.0
            rise = np.nan_to_num(s[f"{st}_stage"].to_numpy() - stage0, nan=0.0) if np.isfinite(stage0) else s[f"{st}_hmax"].to_numpy()
            row[f"{k}_water_arr"] = first(t, (rise > 0.5) | (np.abs(s[f"{st}_Q"].to_numpy()) > q0 + 200))
            row[f"{k}_debris_arr"] = first(t, s[f"{st}_cmax"].to_numpy() > 0.05)
            row[f"{k}_peakQ"] = float(np.abs(s[f"{st}_Q"]).max())
            row[f"{k}_peak_rise"] = float(np.nanmax(rise))
            lo, hi = OBS[k]
            v = row[f"{k}_water_arr"]
            row[f"{k}_ok"] = bool(np.isfinite(v) and lo <= v <= hi)
        row["front_km@463"] = float(np.interp(463, t, s.debris_front_upper_km.fillna(0)))
        row["front_km_end"] = float(np.nanmax(s.debris_front_upper_km))
        row["umax"] = float(s.max_speed_m_s.max())
        row["t_end"] = float(t[-1])
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(df.round(0).to_string())
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
