"""Circularity check for the joint inversion: does refitting the storage section per volume cell rescue a large volume?
Routes one corridor run through a small (h_bank x fp_scale) grid and reports the gauge-group misfit of each routing.
Usage: python swe/storage_refit_check.py --runs vf60_s3_i0.2 vf60_s1.5_i0.2 vf60_s0.75_i0.2"""

from __future__ import annotations

import os

import argparse
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1]))); sys.path.insert(0, str(ROOT / "swe"))
import route1d, sweep  # noqa: E402

UPPER = ("gyirong_cctv_arrival", "rasuwagadhi_signal_loss", "syabrubesi_signal_loss")


def gauge_misfit(run, out):
    s2 = pd.read_csv(ROOT / "runs" / run / "series.csv"); s1 = pd.read_csv(ROOT / "runs" / out / "series.csv"); terms = []
    for st, qty, target, tol in sweep.CRITERIA:
        if st in UPPER:
            continue
        v = sweep.station_metrics(s1, st)[qty]; terms.append(9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0))
    return float(np.mean(terms))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--runs", nargs="+", required=True); a = p.parse_args()
    rows = []
    for run in a.runs:
        for hb in (2.0, 3.0, 4.0):
            for fp in (1.0, 2.0, 3.0):
                out = f"r1d_{run}_b{hb:g}_f{fp:g}_s8"
                if not (ROOT / "runs" / out / "series.csv").exists():
                    args = SimpleNamespace(tag="corridor60s", start="syabrubesi_signal_loss", driver=run, out=out, n_w=0.03, n_d=0.02, dep_uc=0.0, dep_tau=600.0,
                                           width_scale=1.0, h_bank=hb, fp_scale=fp, t_end=28800.0, frame_dt=60.0, cfl=0.5, spin=28800.0, quiet=True)
                    with contextlib.redirect_stdout(io.StringIO()):
                        route1d.run(args)
                rows.append({"run": run, "h_bank": hb, "fp_scale": fp, "gauge_misfit": round(gauge_misfit(run, out), 2)}); print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(ROOT / "figures" / "paper" / "table_storage_refit_check.csv", index=False)
    print(df.pivot_table(index="run", columns=["h_bank", "fp_scale"], values="gauge_misfit").to_string())
    print("best per run:"); print(df.groupby("run").gauge_misfit.min())


if __name__ == "__main__":
    main()
