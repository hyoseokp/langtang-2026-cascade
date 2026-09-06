"""Score an event run against the pre-registered criteria in docs/RESEARCH_PLAN.md (stored data only)."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
# windows in seconds after the USGS onset (02:52:10 UTC)
CRITERIA = {
    "gyirong_cctv_arrival": {"window": (463 - 90, 463 + 90), "nominal": 463},
    "rasuwagadhi_signal_loss": {"window": (170, 770)},
    "syabrubesi_signal_loss": {"window": (770, 1370)},
    "devghat_rise": {"window": (21170, 21770)},
    "devghat_peak": {"window": (26570 - 900, 26570 + 900), "nominal": 26570},
    "devghat_ratio": {"window": (1.7, 2.8), "nominal": 2.25},
    "devghat_excess_mm3": {"window": (10.0, 30.0), "nominal": 19.96, "integration_s": (5 * 3600 + 33 * 60, 9 * 3600 + 23 * 60)},
    # DHM stage records (10-min sampling): rise = first +0.3 m, peak time; amplitude compared separately
    "galchhi_rise": {"window": (1.6 * 3600, 2.2 * 3600), "nominal": 1.88 * 3600},
    "galchhi_peak": {"window": (1.9 * 3600, 2.6 * 3600), "nominal": 2.21 * 3600},
    "kali_khola_rise": {"window": (4.5 * 3600, 5.2 * 3600), "nominal": 4.8 * 3600},
    "kali_khola_peak": {"window": (5.3 * 3600, 6.1 * 3600), "nominal": 5.71 * 3600},
    "betrawati_arrival": {"window": (2570, 3170)},  # telemetry loss 09:20-09:30 NPT
    "malekhu_arrival": {"window": (9000, 10200)},  # warning level 11:26 NPT, 10.48 m at 11:40
}
OBS_STAGE_RISE_M = {"galchhi_gauge": 8.09, "kali_khola_gauge": 4.99, "devghat_2023_candidate": 1.88}
STATIONS = ["gyirong_cctv_arrival", "rasuwagadhi_signal_loss", "syabrubesi_signal_loss", "betrawati_gauge",
            "galchhi_gauge", "malekhu_gauge", "kali_khola_gauge", "devghat_2023_candidate"]


def first_time(t: np.ndarray, cond: np.ndarray) -> float:
    idx = np.nonzero(cond)[0]
    return float(t[idx[0]]) if len(idx) else float("nan")


def footprint_scores(run_id: str, base_run: str, tag: str = "corridor60") -> dict:
    """Maximum simulated inundation/debris footprint vs satellite-mapped surfaces (inside the UNOSAT extent)."""
    obs_path = ROOT / "inputs" / f"{tag}_observed_footprint.npz"
    frames = sorted((ROOT / "runs" / run_id / "frames").glob("frame_*.npz"))
    if not obs_path.exists() or not frames:
        return {}
    obs = np.load(obs_path)
    inp = np.load(ROOT / "inputs" / f"{tag}.npz")
    h_base = np.load(ROOT / "runs" / base_run / "final_state.npz")["h"].ravel()
    n = h_base.size
    max_excess = np.zeros(n, np.float32); max_debris = np.zeros(n, np.float32)
    for fp in frames:
        fr = np.load(fp)
        idx = fr["idx"]; h = fr["h"].astype(np.float32); c = fr["c"].astype(np.float32)
        np.maximum.at(max_excess, idx, h - h_base[idx])
        np.maximum.at(max_debris, idx, h * c)
    shape = inp["z"].shape
    sim = ((max_excess > 0.5) | (max_debris > 0.3)).reshape(shape)
    ext = obs["analysis_extent"]
    route = np.isfinite(inp["route_chainage_m"]) | np.isfinite(inp["upper_chainage_m"])
    cell_km2 = (30.0 if "30" in tag else 60.0) ** 2 / 1e6
    result = {"sim_footprint_km2": float(sim.sum() * cell_km2), "criterion": "max excess depth > 0.5 m or debris depth > 0.3 m"}
    for name in ("unosat_affected", "unosat_floodextent", "emsr927_aoi01", "emsr927_aoi02", "emsr927_aoi03"):
        o = obs[name] & ext
        s = sim & ext
        tp = float((o & s).sum()); fp_ = float((~o & s).sum()); fn = float((o & ~s).sum())
        result[name] = {"obs_km2": float(o.sum() * cell_km2), "precision": tp / max(tp + fp_, 1), "recall": tp / max(tp + fn, 1),
                        "iou": tp / max(tp + fp_ + fn, 1)}
    ch = inp["route_chainage_m"]
    sim_ch = ch[sim & np.isfinite(ch)]
    obs_ch = ch[obs["unosat_affected"] & np.isfinite(ch)]
    result["sim_max_route_chainage_km"] = float(sim_ch.max() / 1e3) if sim_ch.size else None
    result["obs_max_route_chainage_km"] = float(obs_ch.max() / 1e3) if obs_ch.size else None
    np.savez_compressed(ROOT / "runs" / run_id / "max_footprint.npz", max_excess=max_excess.reshape(shape),
                        max_debris=max_debris.reshape(shape), sim=sim)
    return result


def score(run_id: str, base_run: str, tag: str = "corridor60") -> dict:
    v2 = ROOT / "runs" / run_id / "series_transects_v2.csv"
    s = pd.read_csv(v2 if v2.exists() else ROOT / "runs" / run_id / "series.csv")
    bv2 = ROOT / "runs" / base_run / "baseline_transects_v2.csv"
    base = pd.read_csv(bv2).iloc[0] if (v2.exists() and bv2.exists()) else pd.read_csv(ROOT / "runs" / base_run / "series.csv").iloc[-1]
    t = s["time_s"].to_numpy()
    out = {"run_id": run_id, "base_run": base_run, "stations": {}, "criteria": {}}
    for st in STATIONS:
        if f"{st}_Q" not in s.columns:
            continue
        q, q0 = np.abs(s[f"{st}_Q"].to_numpy()), abs(float(base[f"{st}_Q"]))
        hmax, h0 = s[f"{st}_hmax"].to_numpy(), float(base[f"{st}_hmax"])
        stage, stage0 = s[f"{st}_stage"].to_numpy(), float(base[f"{st}_stage"])
        cmax = s[f"{st}_cmax"].to_numpy()
        rise = np.nan_to_num(stage - stage0, nan=0.0)
        arrival_flow = first_time(t, (q > q0 + 0.2 * max(q0, 1.0) + 50.0) | (rise > 0.5) | (cmax > 0.05))
        arrival_debris = first_time(t, cmax > 0.05)
        ipk = int(np.nanargmax(q)) if np.isfinite(q).any() else 0
        out["stations"][st] = {
            "baseline_Q_m3_s": q0, "baseline_hmax_m": h0, "baseline_stage_m": stage0,
            "arrival_s": arrival_flow, "debris_arrival_s": arrival_debris,
            "peak_Q_m3_s": float(q[ipk]), "peak_time_s": float(t[ipk]), "peak_ratio": float(q[ipk] / q0) if q0 > 0 else float("nan"),
            "peak_stage_rise_m": float(np.nanmax(rise)), "peak_cmax": float(np.nanmax(cmax)),
        }
    st = out["stations"]
    dv = st["devghat_2023_candidate"]
    q = np.abs(s["devghat_2023_candidate_Q"].to_numpy()); q0 = dv["baseline_Q_m3_s"]
    t0, t1 = CRITERIA["devghat_excess_mm3"]["integration_s"]
    m = (t >= t0) & (t <= t1)
    excess = float(np.trapezoid(np.maximum(q[m] - q0, 0.0), t[m]) / 1e6) if m.sum() > 1 else float("nan")
    gross = float(np.trapezoid(q[m], t[m]) / 1e6) if m.sum() > 1 else float("nan")
    rise_t = first_time(t, np.nan_to_num(s["devghat_2023_candidate_stage"].to_numpy() - dv["baseline_stage_m"], nan=0) > 0.3)
    def rise_time(key):
        stg = s[f"{key}_stage"].to_numpy(); s0 = st[key]["baseline_stage_m"]
        return first_time(t, np.nan_to_num(stg - s0, nan=0) > 0.3)
    values = {
        "galchhi_rise": rise_time("galchhi_gauge"), "galchhi_peak": st["galchhi_gauge"]["peak_time_s"],
        "kali_khola_rise": rise_time("kali_khola_gauge"), "kali_khola_peak": st["kali_khola_gauge"]["peak_time_s"],
        "gyirong_cctv_arrival": st["gyirong_cctv_arrival"]["arrival_s"],
        "rasuwagadhi_signal_loss": st["rasuwagadhi_signal_loss"]["arrival_s"],
        "syabrubesi_signal_loss": st["syabrubesi_signal_loss"]["arrival_s"],
        "betrawati_arrival": st.get("betrawati_gauge", {}).get("arrival_s", float("nan")),
        "malekhu_arrival": st.get("malekhu_gauge", {}).get("arrival_s", float("nan")),
        "devghat_rise": rise_t, "devghat_peak": dv["peak_time_s"], "devghat_ratio": dv["peak_ratio"],
        "devghat_excess_mm3": excess,
    }
    passed_all = True
    for k, v in values.items():
        lo, hi = CRITERIA[k]["window"]
        ok = bool(np.isfinite(v) and lo <= v <= hi)
        passed_all &= ok
        out["criteria"][k] = {"value": v, "window": [lo, hi], "pass": ok}
    out["devghat_gross_volume_window_mm3"] = gross
    out["stage_rise_model_vs_obs_m"] = {k: [st[k]["peak_stage_rise_m"], v] for k, v in OBS_STAGE_RISE_M.items()}
    out["footprint"] = footprint_scores(run_id, base_run, tag)
    out["consistent_with_all_criteria"] = passed_all
    out["run_end_s"] = float(t[-1])
    out["final_volume_mm3"] = float(s["total_volume_m3"].iloc[-1] / 1e6)
    out["max_debris_front_route_km"] = float(np.nanmax(s["debris_front_route_km"]))
    out["max_flood_front_route_km"] = float(np.nanmax(s["flood_front_route_km"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--base-run", default="spinup60_v3")
    p.add_argument("--tag", default="corridor60")
    a = p.parse_args()
    result = score(a.run_id, a.base_run, a.tag)
    (ROOT / "runs" / a.run_id / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("run_id", "criteria", "consistent_with_all_criteria",
                                             "max_debris_front_route_km", "max_flood_front_route_km")}, indent=1))
    for st, v in result["stations"].items():
        print(f"{st:26s} base Q {v['baseline_Q_m3_s']:8.0f} peak Q {v['peak_Q_m3_s']:8.0f} at {v['peak_time_s']:7.0f}s "
              f"ratio {v['peak_ratio']:.2f} arrival {v['arrival_s']:.0f}s debris {v['debris_arrival_s']:.0f}s rise {v['peak_stage_rise_m']:.1f} m")


if __name__ == "__main__":
    main()
