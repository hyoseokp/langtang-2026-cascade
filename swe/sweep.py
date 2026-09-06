"""Low-dimensional inverse problem: launch a parameter set on gpu-pc, wait, fetch series, score against all criteria.

misfit = mean over criteria of ((model - target) / tolerance)^2 ; a missing arrival counts 9.
Usage:
  python swe/sweep.py plan  --n 20 --prefix sw1            # write sweep/<prefix>_plan.json (Latin hypercube)
  python swe/sweep.py run   --prefix sw1 --concurrent 5    # launch, wait, fetch, score -> sweep/<prefix>_scores.csv
  python swe/sweep.py score --runs fc60s_a fc60s_b          # score existing runs only
"""

from __future__ import annotations

import os

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
PY = sys.executable
GPU = [PY, str(ROOT / "swe" / "gpu_pc.py")]  # remote launcher of the original workspace; "score" mode needs no GPU host

# (station, quantity, target, tolerance) ; quantities: arr_s (stage rise > 0.5 m), peak_s, rise_m, ratio, excess_Mm3
CRITERIA = [
    ("gyirong_cctv_arrival", "arr_s", 463, 90),
    ("rasuwagadhi_signal_loss", "arr_s", 470, 300),
    ("syabrubesi_signal_loss", "arr_s", 1070, 300),
    ("betrawati_gauge", "arr_s", 2870, 300),
    ("galchhi_gauge", "arr_s", 6840, 1080),
    ("galchhi_gauge", "peak_s", 8100, 1260),
    ("galchhi_gauge", "rise_m", 8.8, 1.0),
    ("malekhu_gauge", "arr_s", 9840, 300),
    ("kali_khola_gauge", "arr_s", 17460, 1260),
    ("kali_khola_gauge", "rise_m", 5.6, 0.7),
    ("devghat_2023_candidate", "arr_s", 21510, 300),
    ("devghat_2023_candidate", "peak_s", 26640, 900),
    ("devghat_2023_candidate", "rise_m", 1.9, 0.4),
    ("devghat_2023_candidate", "ratio", 2.25, 0.55),
    ("devghat_2023_candidate", "excess_Mm3", 20.0, 10.0),
]

SPACE = {  # name: (low, high, log?)
    "n_w": (0.02, 0.035, False),
    "release_c0": (0.7, 1.0, False),
    "n_d": (0.012, 0.03, False),
    "dep_uc": (1.5, 6.0, False),
    "dep_tau": (150.0, 2400.0, True),
    "erosion_k": (0.002, 0.01, True),
}
FIXED = ("--restart runs/spinup60s_c/final_state.npz --release --release-ice-fraction 0.25 --release-duration 30 "
         "--melt-eff 1.0 --erosion-uc 6.0 --erodible-depth 5.0 --t-end 28800 --frame-dt 60 --mu-s 0")


def station_metrics(s: pd.DataFrame, st: str) -> dict:
    q = np.abs(s[f"{st}_Q"].to_numpy()); q0 = q[0]; stg = s[f"{st}_stage"].to_numpy(); t = s.time_s.to_numpy()
    i = np.nonzero((stg - stg[0] > 0.5) | (q > 2.0 * q0 + 50.0))[0]; ip = int(np.argmax(q))
    return {"arr_s": float(t[i[0]]) if len(i) else np.nan, "peak_s": float(t[ip]) if (q[ip] > 1.5 * q0 and ip < len(q) - 1) else np.nan,
            "rise_m": float(np.nanmax(stg) - stg[0]), "ratio": float(q[ip] / max(q0, 1.0)),
            "excess_Mm3": float(np.trapezoid(np.maximum(q - q0, 0), t) / 1e6), "peak_Q": float(q[ip])}


def score_run(run: str) -> dict:
    s = pd.read_csv(ROOT / "runs" / run / "series.csv")
    res = json.loads((ROOT / "runs" / run / "result.json").read_text("utf-8"))
    out = {"run": run, "entrained_Mm3": res["entrained_m3"] / 1e6, "deposited_Mm3": res.get("deposited_m3", 0) / 1e6}
    terms = []
    cache = {}
    for st, qty, target, tol in CRITERIA:
        m = cache.setdefault(st, station_metrics(s, st))
        v = m[qty]
        term = 9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0)
        terms.append(term)
        out[f"{st.split('_')[0]}_{qty}"] = round(v, 2) if np.isfinite(v) else np.nan
    out["misfit"] = float(np.mean(terms)); out["n_pass"] = int(sum(t <= 1.0 for t in terms))
    for k, a in res["args"].items():
        if k in ("n_w", "n_d", "dep_uc", "dep_tau", "erosion_k", "release_ice_fraction", "release_c0", "eta0"):
            out[k] = a
    return out


def plan(a):
    rng = np.random.default_rng(a.seed)
    names = list(SPACE); n = a.n
    cols = []
    for name in names:
        lo, hi, lg = SPACE[name]
        u = (rng.permutation(n) + rng.random(n)) / n
        v = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if lg else lo + u * (hi - lo)
        cols.append(v)
    runs = []
    for i in range(n):
        p = {name: float(round(c[i], 4)) for name, c in zip(names, cols)}
        runs.append({"run": f"{a.prefix}_{i:02d}", "params": p})
    (ROOT / "sweep").mkdir(exist_ok=True)
    (ROOT / "sweep" / f"{a.prefix}_plan.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print(f"planned {n} runs -> sweep/{a.prefix}_plan.json")


def sh(args):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def is_done(run):
    return "done {" in sh(GPU + ["tail", "--run-id", run])


def run_sweep(a):
    runs = json.loads((ROOT / "sweep" / f"{a.prefix}_plan.json").read_text("utf-8"))
    pending = [r for r in runs if not (ROOT / "runs" / r["run"] / "series.csv").exists()]
    active: list[str] = []
    while pending or active:
        active = [r for r in active if not is_done(r)]
        while pending and len(active) < a.concurrent:
            r = pending.pop(0)
            args = FIXED + " " + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in r["params"].items())
            sh(GPU + ["launch", "--tag", a.tag, "--solver", "solver_sparse.py", "--run-id", r["run"], "--args", args])
            active.append(r["run"]); print("launched", r["run"], r["params"], flush=True)
        time.sleep(60)
    rows = []
    for r in runs:
        if not (ROOT / "runs" / r["run"] / "series.csv").exists():
            sh(GPU + ["fetch", "--run-id", r["run"]])
        rows.append(score_run(r["run"]))
    df = pd.DataFrame(rows).sort_values("misfit")
    df.to_csv(ROOT / "sweep" / f"{a.prefix}_scores.csv", index=False)
    print(df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument("action", choices=["plan", "run", "score"])
    p.add_argument("--prefix", default="sw1"); p.add_argument("--n", type=int, default=20); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--concurrent", type=int, default=5); p.add_argument("--tag", default="corridor60s"); p.add_argument("--runs", nargs="*")
    a = p.parse_args()
    if a.action == "plan":
        plan(a)
    elif a.action == "run":
        run_sweep(a)
    else:
        df = pd.DataFrame([score_run(r) for r in a.runs]).sort_values("misfit")
        pd.set_option("display.width", 250); print(df.to_string(index=False))


if __name__ == "__main__":
    main()
