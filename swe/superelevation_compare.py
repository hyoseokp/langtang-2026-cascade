"""Compare modelled maximum flow speed along the upper corridor with Geo-PERA superelevation velocities."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
SRC = Path("C:/Users/연구실/nepal_langtang_2026_inverse_debrisflow/data_raw/terrain/geopera_stereo_v1_1_20260906/repository/sim/inputs")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--run", default="u30f_mu0"); p.add_argument("--tag", default="upper30f")
    a = p.parse_args()
    import sys; sys.path.insert(0, str(ROOT / "swe"))
    from render_profile import build_profile
    inp = np.load(ROOT / "inputs" / f"{a.tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{a.tag}.json").read_text("utf-8"))
    z = inp["z"].astype(np.float32)
    chain, bed, nb, stations, join_km = build_profile(meta, z, inp=inp)
    n = z.size
    vmax = np.zeros(n, np.float32); hmax = np.zeros(n, np.float32)
    for fp in sorted((ROOT / "runs" / a.run / "frames").glob("frame_*.npz")):
        fr = np.load(fp); idx = fr["idx"]
        np.maximum.at(vmax, idx, fr["speed"].astype(np.float32)); np.maximum.at(hmax, idx, fr["h"].astype(np.float32))
    v_prof = vmax[nb].max(axis=1); h_prof = hmax[nb].max(axis=1)
    obs = pd.read_csv(SRC / "superelevation_velocities.csv")
    fl = obs["flags"].fillna("-").astype(str).str.strip(); ok = obs[(fl == "-")].dropna(subset=["v_ms"]).copy()
    # producer chainage origin (354206, 3128245) is ~0.3 km from our source cell; treat as the same origin
    rows = []
    for _, r in ok.iterrows():
        km = r.chainage_m / 1e3
        m = (chain >= km - 0.5) & (chain <= km + 0.5)
        rows.append({"chainage_km": km, "obs_v": r.v_ms, "obs_lo": r.v_lo, "obs_hi": r.v_hi, "obs_W": r.W_m, "obs_dh": r.dh_m,
                     "model_vmax": float(v_prof[m].max()) if m.any() else np.nan, "model_hmax": float(h_prof[m].max()) if m.any() else np.nan})
    df = pd.DataFrame(rows); out = ROOT / "figures" / "paper"; out.mkdir(exist_ok=True, parents=True)
    df.to_csv(out / f"table_superelevation_{a.run}.csv", index=False)
    print(df.round(1).to_string(index=False))
    inside = ((df.model_vmax >= df.obs_lo) & (df.model_vmax <= df.obs_hi)).mean()
    print(f"model vmax within producer range: {inside*100:.0f}% of sites; median ratio model/obs {np.median(df.model_vmax/df.obs_v):.2f}")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(chain, v_prof, color="tab:blue", lw=1, label=f"model maximum speed ({a.run})")
    ax.errorbar(df.chainage_km, df.obs_v, yerr=[df.obs_v - df.obs_lo, df.obs_hi - df.obs_v], fmt="o", color="k", ms=4, capsize=2, label="superelevation velocity (Geo-PERA, accepted)")
    for k, s in stations.items():
        ax.axvline(s["chainage_km"], ls=":", color="grey"); ax.text(s["chainage_km"], ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 80, s["label"], rotation=90, va="top", fontsize=7)
    ax.set_xlabel("chainage from source (km)"); ax.set_ylabel("speed (m/s)"); ax.set_xlim(0, 46); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fig_superelevation_{a.run}.png"); plt.close(fig)


if __name__ == "__main__":
    main()
