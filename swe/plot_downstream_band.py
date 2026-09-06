"""Hydrograph figure: exact 2-D corridor at the upper sites, routed ensemble band at the lower sites, DHM stage rise overlaid.

Reads stored series only (runs/<2d>/series.csv, the 1-D sweep score tables and their runs) and writes
figures/paper/fig_hydrographs.png plus table_downstream_ensemble.csv (ranges over the ensemble).
"""

from __future__ import annotations

import os

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "swe"))
from make_figures import load_dhm, DHM_NAMES  # noqa: E402
import sweep  # noqa: E402

UPPER = [("gyirong_cctv_arrival", "Gyirong Port (21 km)", (373, 553)), ("rasuwagadhi_signal_loss", "Rasuwagadhi (22 km)", (170, 770)),
         ("syabrubesi_signal_loss", "Syabrubesi (36 km)", (770, 1370)), ("betrawati_gauge", "Betrawati (46 km)", (2570, 3170))]
LOWER = [("galchhi_gauge", "Galchhi (80 km)", (5760, 7920), 8.8), ("malekhu_gauge", "Malekhu (105 km)", (9540, 10140), 3.5),
         ("kali_khola_gauge", "Kali Khola (148 km)", (16200, 18720), 5.6), ("devghat_2023_candidate", "Devghat (180 km)", (21240, 21780), 1.9)]


def ensemble(driver: str, tol: float, prefix: str = "fp2"):
    """Members: routings within `tol` of the best misfit on the Galchhi + Devghat fit criteria (hold-out stations unused)."""
    df = pd.read_csv(ROOT / "figures" / "paper" / "table_holdout_routing.csv").sort_values("misfit_fit").reset_index(drop=True)
    df["misfit"] = df["misfit_fit"]; df["n_pass"] = df["pass_fit"] + df["pass_pred"]
    keep = df[df.misfit_fit <= df.misfit_fit.iloc[0] + tol]
    return df, keep


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--two-d", default="prod60"); p.add_argument("--driver", default="sw1_00")
    p.add_argument("--tol", type=float, default=1.0); p.add_argument("--prefix", default="fp2")
    a = p.parse_args()
    df, keep = ensemble(a.driver, a.tol, a.prefix)
    print(f"ensemble: {len(keep)} of {len(df)} routed runs within fit-misfit {df.misfit.iloc[0]:.2f}+{a.tol} (Galchhi+Devghat criteria)")
    print(keep[["run", "misfit", "n_pass"]].to_string(index=False))
    s2 = pd.read_csv(ROOT / "runs" / a.two_d / "series.csv"); t2 = s2.time_s.to_numpy() / 3600
    members = [pd.read_csv(ROOT / "runs" / r / "series.csv") for r in keep.run]
    dhm = load_dhm()
    fig, axes = plt.subplots(4, 2, figsize=(12, 12.5), dpi=150)
    for ax, (key, label, win) in zip(axes[:, 0], UPPER):
        ax.plot(t2, np.abs(s2[f"{key}_Q"]) / 1e3, color="tab:red", lw=1.4, label="2-D corridor model")
        ax.axvspan(win[0] / 3600, win[1] / 3600, color="gold", alpha=0.35, label="observed window")
        ax.set_title(label, fontsize=10); ax.set_ylabel("discharge (10$^3$ m$^3$/s)", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1.0 if key != "betrawati_gauge" else 2.5); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    rows = []
    for ax, (key, label, win, rise_obs) in zip(axes[:, 1], LOWER):
        t = members[0].time_s.to_numpy() / 3600
        R = np.array([m[f"{key}_stage"].to_numpy() - m[f"{key}_stage"].to_numpy()[0] for m in members])
        ax.fill_between(t, R.min(0), R.max(0), color="tab:red", alpha=0.25, lw=0, label=f"routed ensemble ({len(members)} runs)")
        ax.plot(t, np.median(R, 0), color="tab:red", lw=1.2, label="ensemble median")
        ax.axvspan(win[0] / 3600, win[1] / 3600, color="gold", alpha=0.35, label="observed rise window")
        name = DHM_NAMES.get(key)
        if name:
            d = dhm[(dhm.station == name) & (dhm.t_s > -7200) & (dhm.t_s < 8 * 3600)]
            if len(d):
                pre = d[d.t_s <= 0].value_m.median() if (d.t_s <= 0).any() else d.value_m.iloc[0]
                ax.plot(d.t_s / 3600, d.value_m - pre, "k.", ms=3.5, label="DHM stage rise")
        else:
            ax.axhline(rise_obs, color="k", ls="--", lw=0.8, label="reading above warning level")
        ax.set_title(label, fontsize=10); ax.set_ylabel("stage rise (m)", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlim(0, 8); ax.grid(alpha=0.3)
        mets = [sweep.station_metrics(m, key) for m in members]
        rows.append({"station": key, **{q: f"{np.nanmin([m[q] for m in mets]):.2f}-{np.nanmax([m[q] for m in mets]):.2f}" for q in ("arr_s", "peak_s", "rise_m", "ratio", "excess_Mm3")}})
    axes[0, 1].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time after onset (h)", fontsize=8)
    fig.suptitle("Reconstruction against the record: 2-D corridor model (left) and routed ensemble below Syabrubesi (right)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_hydrographs.png"); plt.close(fig)
    st = [sweep.station_metrics(m, "galchhi_gauge")["excess_Mm3"] - sweep.station_metrics(m, "devghat_2023_candidate")["excess_Mm3"] for m in members]
    print(f"storage (routed Galchhi excess - Devghat excess) over members: {min(st):.1f}-{max(st):.1f} Mm3; best member {st[0]:.1f}")
    tab = pd.DataFrame(rows); tab.to_csv(ROOT / "figures" / "paper" / "table_downstream_ensemble.csv", index=False)
    pd.set_option("display.width", 250); print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
