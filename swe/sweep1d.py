"""Grid sweep of the 1-D router; scores the downstream criteria with the same misfit as sweep.py."""
import os
import itertools, sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1]))); sys.path.insert(0, str(ROOT / "swe"))
import route1d, sweep
from types import SimpleNamespace
driver = sys.argv[1] if len(sys.argv) > 1 else "sw1_00"
mode = sys.argv[2] if len(sys.argv) > 2 else "width"
shard, nshard = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
if mode == "fp2":
    grid = {"n_w": [0.03, 0.04], "h_bank": [1.0, 2.0, 3.0, 4.0, 6.0], "fp_scale": [0.5, 1.0, 1.5, 2.0, 3.0], "dep_uc": [0.0]}
elif mode == "fp":
    grid = {"n_w": [0.03, 0.04], "h_bank": [2.0, 4.0, 8.0], "fp_scale": [1.0, 2.0, 4.0], "dep_uc": [0.0, 3.0]}
else:
    grid = {"n_w": [0.03, 0.04, 0.05, 0.06], "width_scale": [1.0, 1.5, 2.0], "dep_uc": [0.0, 3.0]}
rows = []
for ii, vals in enumerate(itertools.product(*grid.values())):
    if ii % nshard != shard: continue
    g = dict(zip(grid.keys(), vals)); nw = g["n_w"]; uc = g["dep_uc"]
    ws = g.get("width_scale", 1.0); hbk = g.get("h_bank", 1e9); fps = g.get("fp_scale", 0.0)
    out = f"r1d_{driver}_n{nw:.3f}_w{ws:.1f}_b{hbk if hbk < 1e8 else 0:g}_f{fps:g}_d{uc:.0f}" + ("_s8" if mode == "fp2" else "")
    a = SimpleNamespace(tag="corridor60s", start="syabrubesi_signal_loss", driver=driver, out=out, n_w=nw, n_d=0.02, dep_uc=uc, dep_tau=600.0,
                        width_scale=ws, h_bank=hbk, fp_scale=fps, t_end=28800.0, frame_dt=60.0, cfl=0.5, spin=(28800.0 if mode == "fp2" else 7200.0), quiet=True)
    import io, contextlib
    if not (ROOT / "runs" / out / "series.csv").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            route1d.run(a)
    s = pd.read_csv(ROOT / "runs" / out / "series.csv"); terms = []; rec = dict(g)
    for st, qty, target, tol in sweep.CRITERIA:
        if f"{st}_Q" not in s.columns: continue
        v = sweep.station_metrics(s, st)[qty]; terms.append(9.0 if not np.isfinite(v) else min(((v - target) / tol) ** 2, 9.0))
        rec[f"{st.split('_')[0]}_{qty}"] = round(v, 2) if np.isfinite(v) else np.nan
    rec["misfit"] = float(np.mean(terms)); rec["n_pass"] = int(sum(t <= 1 for t in terms)); rows.append(rec); print(out, "misfit", round(rec["misfit"], 2), rec["n_pass"], flush=True)
df = pd.DataFrame(rows).sort_values("misfit"); (ROOT / "sweep").mkdir(exist_ok=True); df.to_csv(ROOT / "sweep" / f"sw1d_{driver}_{mode}_scores_{shard}.csv", index=False)
pd.set_option("display.width", 300); print(df.head(12).to_string(index=False))
