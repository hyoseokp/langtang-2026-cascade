"""Paper figures from stored run data only (series.csv, score.json, max_footprint.npz, inputs)."""

from __future__ import annotations

import os

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
UP = ROOT / "data"
ONSET = datetime(2026, 8, 26, 2, 52, 10, tzinfo=timezone.utc)
STATIONS = [("gyirong_cctv_arrival", "Gyirong Port (CCTV proxy)"), ("rasuwagadhi_signal_loss", "Rasuwagadhi gauge"),
            ("syabrubesi_signal_loss", "Syabrubesi gauge"), ("betrawati_gauge", "Betrawati gauge"), ("galchhi_gauge", "Galchhi gauge"),
            ("malekhu_gauge", "Malekhu gauge"), ("kali_khola_gauge", "Kali Khola gauge"), ("devghat_2023_candidate", "Devghat (candidate)")]
WINDOWS = {"gyirong_cctv_arrival": (373, 553), "rasuwagadhi_signal_loss": (170, 770), "syabrubesi_signal_loss": (770, 1370),
           "betrawati_gauge": (2570, 3170), "malekhu_gauge": (9000, 10200), "galchhi_gauge": (6800, 7900), "kali_khola_gauge": (17300, 18700),
           "devghat_2023_candidate": (21170, 21770)}
DHM_NAMES = {"rasuwagadhi_signal_loss": "rasuwagadhi", "syabrubesi_signal_loss": "syabrubesi", "galchhi_gauge": "galchhi",
             "kali_khola_gauge": "kali_khola", "devghat_2023_candidate": "devghat"}


def load_dhm() -> pd.DataFrame:
    frames = []
    for p in (UP / "data_raw/observations/dhm_event_levels.csv", UP / "data_processed/hydrology/dhm_lower_event_levels.csv"):
        d = pd.read_csv(p)
        d["t_s"] = (pd.to_datetime(d["datetime_utc"], utc=True) - ONSET).dt.total_seconds()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fig_hydrographs(runs: list[str], base_run: str, out: Path, t_max_h: float = 8.0) -> None:
    dhm = load_dhm()
    bv2 = ROOT / "runs" / base_run / "baseline_transects_v2.csv"
    base = pd.read_csv(bv2).iloc[0] if bv2.exists() else pd.read_csv(ROOT / "runs" / base_run / "series.csv").iloc[-1]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), dpi=150, sharex=False)
    for ax, (key, label) in zip(axes.ravel(), STATIONS):
        for run in runs:
            v2 = ROOT / "runs" / run / "series_transects_v2.csv"
            s = pd.read_csv(v2 if v2.exists() else ROOT / "runs" / run / "series.csv")
            if f"{key}_Q" not in s.columns:
                continue
            ax.plot(s["time_s"] / 3600, np.abs(s[f"{key}_Q"]), label=run, linewidth=1.2)
        if f"{key}_Q" in base.index:
            ax.axhline(abs(base[f"{key}_Q"]), color="grey", linestyle=":", linewidth=1, label="pre-event baseflow")
        if key in WINDOWS:
            lo, hi = WINDOWS[key]
            ax.axvspan(lo / 3600, hi / 3600, color="gold", alpha=0.35, label="observed window")
        name = DHM_NAMES.get(key)
        if name:
            d = dhm[(dhm.station == name) & (dhm.t_s > -3600) & (dhm.t_s < t_max_h * 3600)]
            if len(d):
                ax2 = ax.twinx()
                ax2.plot(d.t_s / 3600, d.value_m, "k.", markersize=3, label="DHM stage (native datum)")
                ax2.set_ylabel("DHM stage (m)", fontsize=8)
                ax2.tick_params(labelsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("time after onset (h)", fontsize=8)
        ax.set_ylabel("mixture discharge through transect (m3/s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.2, t_max_h)
        if key in ("gyirong_cctv_arrival", "rasuwagadhi_signal_loss", "syabrubesi_signal_loss"):
            ax.set_xlim(-0.05, 1.0)
        if key == "betrawati_gauge":
            ax.set_xlim(-0.05, 3.0)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Transect discharge at the observation sites (model) with observation windows and DHM stage", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_hydrographs.png"); plt.close(fig)


def fig_arrival_table(runs: list[str], out: Path) -> None:
    rows = []
    for run in runs:
        sc = json.loads((ROOT / "runs" / run / "score.json").read_text("utf-8"))
        r = {"run": run}
        for k, v in sc["criteria"].items():
            r[k] = f"{v['value']:.0f}" if k not in ("devghat_ratio", "devghat_excess_mm3") else f"{v['value']:.2f}"
            r[k] += " ok" if v["pass"] else " x"
        r["all"] = sc["consistent_with_all_criteria"]
        r["debris_front_km"] = f"{sc['max_debris_front_route_km']:.1f}"
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_criteria.csv", index=False)
    print(df.to_string())


def fig_footprint(run: str, out: Path, tag: str = "corridor60") -> None:
    inp = np.load(ROOT / "inputs" / f"{tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    obs = np.load(ROOT / "inputs" / f"{tag}_observed_footprint.npz")
    mf = np.load(ROOT / "runs" / run / "max_footprint.npz")
    z = inp["z_raw"]; left, right, bottom, top = meta["extent_utm45n"]
    ls = LightSource(315, 45)
    shade = ls.shade(z, cmap=plt.get_cmap("gray"), vert_exag=1, dx=60, dy=60, vmin=np.percentile(z, 1), vmax=np.percentile(z, 99.5), blend_mode="soft")[..., :3]
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    ax.imshow(shade, extent=(left, right, bottom, top))
    o = obs["unosat_affected"]; s = mf["sim"]
    rgba = np.zeros(z.shape + (4,), np.float32)
    rgba[o & s] = (0.1, 0.8, 0.1, 0.9); rgba[o & ~s] = (0.1, 0.3, 1.0, 0.9); rgba[~o & s] = (1.0, 0.3, 0.1, 0.9)
    ax.imshow(rgba, extent=(left, right, bottom, top), interpolation="nearest")
    for k, st in meta["stations"].items():
        ax.plot(st["x"], st["y"], "w^", markeredgecolor="k", markersize=7)
        ax.annotate(st["label"], (st["x"], st["y"]), xytext=(5, 5), textcoords="offset points", fontsize=8, color="white")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=(0.1, 0.8, 0.1), label="mapped and simulated"), Patch(color=(0.1, 0.3, 1.0), label="mapped only (missed)"),
                       Patch(color=(1.0, 0.3, 0.1), label="simulated only")], loc="lower left", fontsize=9)
    ax.set_title(f"Maximum simulated footprint ({run}) vs UNOSAT mapped affected surface", fontsize=11)
    ax.set_xlabel("UTM 45N easting (m)"); ax.set_ylabel("northing (m)")
    fig.tight_layout(); fig.savefig(out / f"fig_footprint_{run}.png"); plt.close(fig)


def fig_ledger(runs: list[str], out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for run in runs:
        s = pd.read_csv(ROOT / "runs" / run / "series.csv")
        ax[0].plot(s.time_s / 3600, s.total_volume_m3 / 1e6, label=f"{run} total")
        ax[0].plot(s.time_s / 3600, s.debris_volume_m3 / 1e6, "--", label=f"{run} debris")
        ax[1].plot(s.time_s / 3600, s.debris_front_route_km, label=f"{run} debris front")
        ax[1].plot(s.time_s / 3600, s.flood_front_route_km, ":", label=f"{run} flood front")
    ax[0].set_xlabel("time after onset (h)"); ax[0].set_ylabel("mobile volume in domain (Mm3)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("time after onset (h)"); ax[1].set_ylabel("front chainage along mapped route (km)"); ax[1].legend(fontsize=7)
    for k, (lo, hi) in {"Gyirong": (463, 463), "Syabrubesi": (770, 1370), "Devghat": (21170, 21770)}.items():
        pass
    fig.tight_layout(); fig.savefig(out / "fig_ledger.png"); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--base-run", default="spinup60_v6b")
    p.add_argument("--footprint-run")
    a = p.parse_args()
    out = ROOT / "figures" / "paper"; out.mkdir(parents=True, exist_ok=True)
    fig_hydrographs(a.runs, a.base_run, out)
    fig_arrival_table(a.runs, out)
    fig_ledger(a.runs, out)
    if a.footprint_run:
        fig_footprint(a.footprint_run, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
