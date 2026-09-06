"""1-D longitudinal-profile animation along the corridor (bed slope + water surface) from stored frames."""

from __future__ import annotations

import os

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import imageio_ffmpeg
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import shape

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"; matplotlib.rcParams["font.weight"] = "normal"
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
UP = ROOT / "data"
NPT = timezone(timedelta(hours=5, minutes=45))
ONSET_UTC = datetime(2026, 8, 26, 2, 52, 10, tzinfo=timezone.utc)
DEBRIS_RGB = np.array([0.80, 0.30, 0.02]); WATER_RGB = np.array([0.05, 0.40, 0.80])
OBS = {"gyirong_cctv_arrival": (373, 553), "rasuwagadhi_signal_loss": (170, 770), "syabrubesi_signal_loss": (770, 1370),
       "devghat_2023_candidate": (21170, 21770)}
_CTX: dict = {}


def build_profile(meta: dict, z: np.ndarray, step_m: float = 60.0, inp: dict | None = None):
    """Longitudinal profile along the actual D8 channel path from the source (monotone bed)."""
    if inp is not None and "rcv" in inp:
        return build_profile_d8(meta, z, inp)
    return build_profile_lines(meta, z, step_m)


def build_profile_d8(meta: dict, z: np.ndarray, inp: dict):
    rows, cols = z.shape; cell = meta["cell_m"]
    rcv = inp["rcv"].astype(np.int64).ravel(); src = inp["source_mask"]
    r0, c0 = np.argwhere(src)[np.argmin(z[src])]
    node = int(r0 * cols + c0); path = [node]
    while rcv[node] >= 0:
        node = int(rcv[node]); path.append(node)
    path = np.array(path); pr, pc = np.divmod(path, cols)
    step = np.where((np.diff(pr) != 0) & (np.diff(pc) != 0), cell * np.sqrt(2.0), cell)
    chain = np.r_[0.0, np.cumsum(step)]
    nb = np.stack([np.clip(pr + dr, 0, rows - 1) * cols + np.clip(pc + dc, 0, cols - 1) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=1)
    bed = z.ravel()[path]
    left, right, bottom, top = meta["extent_utm45n"]
    px = left + (pc + 0.5) * cell; py = top - (pr + 0.5) * cell
    stations = {}
    join_km = 0.0
    for key, s in meta["stations"].items():
        if key == "usgs_main_onset":
            continue
        i = int(np.argmin((px - s["x"]) ** 2 + (py - s["y"]) ** 2))
        stations[key] = {"label": s["label"], "chainage_km": chain[i] / 1e3}
        if key == "rasuwagadhi_signal_loss":
            join_km = chain[i] / 1e3
    return chain / 1e3, bed, nb, stations, join_km


def build_profile_lines(meta: dict, z: np.ndarray, step_m: float = 60.0):
    """Source -> Rasuwagadhi along the observed centreline, then the mapped route to Devghat."""
    route = gpd.read_file(UP / "data_processed/terrain/downstream_route_osm_candidate.geojson").to_crs(32645).geometry.iloc[0]
    center = shape(json.loads((UP / "data_processed/terrain/observed_corridor_centerline.geojson").read_text("utf-8"))["features"][0]["geometry"])
    join_on_center = center.project(route.interpolate(0.0))
    pts, chain = [], []
    d = 0.0
    while d <= join_on_center:
        pts.append(center.interpolate(d).coords[0]); chain.append(d); d += step_m
    d = 0.0
    while d <= route.length:
        pts.append(route.interpolate(d).coords[0]); chain.append(join_on_center + d); d += step_m
    pts = np.array(pts); chain = np.array(chain)
    left, right, bottom, top = meta["extent_utm45n"]; cell = meta["cell_m"]
    rows = ((top - pts[:, 1]) // cell).astype(int); cols = ((pts[:, 0] - left) // cell).astype(int)
    ok = (rows >= 1) & (rows < z.shape[0] - 1) & (cols >= 1) & (cols < z.shape[1] - 1)
    rows, cols, chain, pts = rows[ok], cols[ok], chain[ok], pts[ok]
    # 3x3 neighbourhood indices for a thalweg-tolerant sample
    nb = np.stack([(rows + dr) * z.shape[1] + (cols + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=1)
    bed = z.ravel()[nb].min(axis=1)
    stations = {}
    for key, s in meta["stations"].items():
        if key == "usgs_main_onset":
            continue
        i = int(np.argmin((pts[:, 0] - s["x"]) ** 2 + (pts[:, 1] - s["y"]) ** 2))
        stations[key] = {"label": s["label"], "chainage_km": chain[i] / 1e3}
    return chain / 1e3, bed, nb, stations, join_on_center / 1e3


def sample(frame, nb, n_cells):
    h = np.zeros(n_cells, np.float32); c = np.zeros(n_cells, np.float32)
    h[frame["idx"]] = frame["h"].astype(np.float32); c[frame["idx"]] = frame["c"].astype(np.float32)
    hn = h[nb]; k = hn.argmax(axis=1)
    dz = np.zeros(n_cells, np.float32)
    if "dz_idx" in frame:
        dz[frame["dz_idx"]] = frame["dz"].astype(np.float32)
    dzn = dz[nb]
    return hn.max(axis=1), c[nb][np.arange(len(k)), k], np.minimum(dzn[np.arange(len(k)), k], 0.0), np.median(dzn, axis=1).clip(min=0.0)


def _init(ctx):
    _CTX.update(ctx)


def render_one(args):
    i, path = args
    ctx = _CTX
    fr = np.load(path); t = float(fr["time_s"])
    h, c, dz, dep = sample(fr, ctx["nb"], ctx["n_cells"])  # dz: thalweg scour (<=0); dep: median deposit over the 3x3 (>=0; single-cell piles do not spike)
    if ctx.get("smooth", 0) > 1:
        from scipy.ndimage import median_filter
        k = ctx["smooth"]; h = median_filter(h, size=k, mode="nearest"); c = median_filter(c, size=k, mode="nearest"); dep = median_filter(dep, size=k, mode="nearest"); dz = median_filter(dz, size=k, mode="nearest")
    x, bed0, hb = ctx["chain"], ctx["bed"], ctx["h_base"]
    r = ctx.get("r1d")
    if r is not None:  # below the routing start station: the 1-D compound-section routing replaces the 2-D corridor state
        sel, s_at = r["sel"], r["s_at"]; k = int(np.argmin(np.abs(r["t"] - t)))
        h = h.copy(); c = c.copy(); dep = dep.copy(); dz = dz.copy()
        h[sel] = np.interp(s_at, r["s"], r["h"][k]); c[sel] = np.interp(s_at, r["s"], r["c"][k]); dep[sel] = 0.0; dz[sel] = 0.0
    ex = ctx["exag"]
    bed = bed0 + dz * ex  # scoured thalweg (exaggerated like the flow)
    clock = (ONSET_UTC + timedelta(seconds=t)).astimezone(NPT)
    clean = ctx.get("clean", False); fs = 28 if clean else 8
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8.5), dpi=ctx["dpi"], gridspec_kw={"height_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.1 if clean else 0.07, right=0.98, top=0.97 if clean else 0.9, bottom=0.14 if clean else 0.08, hspace=0.3 if clean else 0.25)
    ax1.fill_between(x, ctx["zmin"], bed, color="#6b6b6b", lw=0)
    ax1.fill_between(x, bed, bed + dep * ex, where=dep > 0, facecolor=(0.3, 0.16, 0.05), edgecolor="k", hatch="///", lw=0, label=f"deposited solids, valley floor (x{ex:g})")
    rise = np.maximum(h + dz - hb, 0.0)  # water-surface rise above the pre-event level (true scale; dz = thalweg scour)
    hb_d = np.minimum(hb, 8.0)  # DSM pit ponds are clipped in the display
    ax1.fill_between(x, bed, bed + hb_d * ex, color=(0.55, 0.65, 0.75), lw=0, label=f"pre-event river (x{ex:g}; pit ponds clipped at 8 m)")
    surf = bed + np.minimum(h, hb_d + rise) * ex
    pts = np.array([x, surf]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    w = np.clip(c[:-1, None] / 0.35, 0.0, 1.0)
    cols = WATER_RGB[None, :] * (1 - w) + DEBRIS_RGB[None, :] * w
    ax1.fill_between(x, bed + hb_d * ex, np.maximum(surf, bed + hb_d * ex), color=(0.9, 0.55, 0.3), alpha=0.35, lw=0, label="flow (colour: solids fraction)")
    ax1.add_collection(LineCollection(segs, colors=cols, linewidths=2.0))
    ax1.set_xlim(x[0], x[-1]); ax1.set_ylim(ctx["zmin"], ctx["zmax"])
    ax1.set_ylabel(("elevation (m)" if clean else "elevation (m, EGM2008); flow depth exaggerated x%g" % ex), fontsize=fs); ax1.tick_params(labelsize=fs - 2)
    for key, s in ctx["stations"].items():
        ax1.axvline(s["chainage_km"], color="k", lw=0.6, ls=":")
        lab = s["label"].replace(" CCTV", "").replace(" (candidate)", "") if clean else s["label"]
        if clean and key == "rasuwagadhi_signal_loss":
            continue  # 1 km from Gyirong: one label serves both on the paper panels
        if clean:  # paper panels: label inside the bed fill, reading upward from the bottom
            ax1.text(s["chainage_km"], ctx["zmin"] + 0.04 * (ctx["zmax"] - ctx["zmin"]), lab, rotation=90, va="bottom", ha="right",
                     fontsize=fs - 8, color="black", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
        else:
            ax1.text(s["chainage_km"], ctx["zmax"] - 0.04 * (ctx["zmax"] - ctx["zmin"]), lab, rotation=90, va="top", ha="right", fontsize=fs)
    ax1.axvline(ctx["join_km"], color="grey", lw=0.5)
    if r is not None:
        x0 = x[r["sel"]][0]; ax1.axvline(x0, color="k", lw=0.8, ls="--"); ax2.axvline(x0, color="k", lw=0.8, ls="--")
        if not clean:
            ax1.text(x0 + 1, ctx["zmin"] + 0.06 * (ctx["zmax"] - ctx["zmin"]), "below: 1-D routing, compound section (channel + valley-floor storage)", fontsize=7)
    if not clean:
        ax1.set_title(f"{ctx['label']}\nt = {t:.0f} s ({t/3600:.2f} h) | {clock:%H:%M:%S} NPT | longitudinal profile: source -> Rasuwagadhi (observed corridor) -> Devghat (mapped route)", fontsize=10)
        ax1.legend(loc="upper right", fontsize=8)
    ax2.fill_between(x, 0, rise, color=(0.6, 0.6, 0.6), lw=0, label="rise above pre-event level (true scale)")
    ax2.fill_between(x, 0, np.minimum(h * c, rise), color=DEBRIS_RGB, lw=0, label="of which solids, h*c")
    ax2.fill_between(x, 0, dep, facecolor=(0.3, 0.16, 0.05), edgecolor="k", hatch="///", lw=0, label="deposit thickness (settled, valley floor)")
    ax2.plot(x, -dz, color=(0.3, 0.2, 0.1), lw=0.6, label="channel scour depth")
    ax2.set_ylim(0, ctx["hmax"]); ax2.set_xlim(x[0], x[-1])
    ax2.set_xlabel("chainage along corridor (km)", fontsize=fs); ax2.set_ylabel("rise (m)" if clean else "rise above pre-event level (m)", fontsize=fs); ax2.tick_params(labelsize=fs - 2)
    for key, s in ctx["stations"].items():
        ax2.axvline(s["chainage_km"], color="k", lw=0.6, ls=":")
        if key in OBS and not clean:
            lo, hi = OBS[key]
            state = "expected now" if lo <= t <= hi else ("window passed" if t > hi else "window ahead")
            ax2.text(s["chainage_km"], ctx["hmax"] * 0.95, f"obs {lo}-{hi} s\n{state}", fontsize=7, ha="left", va="top",
                     color="darkred" if state == "expected now" else "black")
    if not clean:
        ax2.legend(loc="upper right", fontsize=8)
    out = ctx["out"] / f"p_{i:05d}.png"
    fig.savefig(out); plt.close(fig)
    return i


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True); p.add_argument("--base-run", required=True)
    p.add_argument("--tag", default="corridor60"); p.add_argument("--exag", type=float, default=15.0)
    p.add_argument("--label", default="INDEPENDENT MIXTURE-SWE RECONSTRUCTION (diagnostic)")
    p.add_argument("--states-per-second", type=float, default=6.0); p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=90); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--xmax", type=float, default=0.0, help="cut the profile at this chainage (km); 0 = full")
    p.add_argument("--smooth", type=int, default=0, help="median filter (cells) applied along the profile to depths for display (DSM pit artefacts)")
    p.add_argument("--route1d", default="", help="run id of a route1d routing (profile.npz) shown below its start station instead of the 2-D state")
    p.add_argument("--paper-frames", default="", help="comma-separated frame indices: render only these as clean paper panels (no video)")
    p.add_argument("--hmax", type=float, default=0.0, help="fixed upper limit of the rise panel (m); 0 = from the frames")
    a = p.parse_args()
    inp = np.load(ROOT / "inputs" / f"{a.tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{a.tag}.json").read_text("utf-8"))
    z = inp["z"].astype(np.float32)
    chain, bed, nb, stations, join_km = build_profile(meta, z, inp=inp)
    if a.xmax > 0:
        keep = chain <= a.xmax; chain, bed, nb = chain[keep], bed[keep], nb[keep]
        stations = {k: v for k, v in stations.items() if v["chainage_km"] <= a.xmax}
    h_base_full = np.load(ROOT / "runs" / a.base_run / "final_state.npz")["h"].astype(np.float32).ravel()
    hb = h_base_full[nb].max(axis=1)
    if a.smooth > 1:
        from scipy.ndimage import median_filter
        hb = median_filter(hb, size=a.smooth, mode="nearest")
    r1d = None
    if a.route1d:
        pr = np.load(ROOT / "runs" / a.route1d / "profile.npz"); res = json.loads((ROOT / "runs" / a.route1d / "result.json").read_text("utf-8"))
        start = res["args"]["start"]; skm = res["stations_km"]
        keys = [k for k in skm if k in stations]; xs = np.array([stations[start]["chainage_km"]] + [stations[k]["chainage_km"] for k in keys]); ss = np.array([0.0] + [skm[k] * 1e3 for k in keys])
        o = np.argsort(xs); xs, ss = xs[o], ss[o]
        sel = chain >= stations[start]["chainage_km"]; s_at = np.maximum(np.interp(chain[sel], xs, ss), 660.0)  # skip the mass-source cells of the 1-D model
        r1d = {"sel": sel, "s_at": s_at, "s": pr["s_m"], "t": pr["time_s"], "h": pr["h"], "c": pr["c"]}
        hb = hb.copy(); hb[sel] = np.interp(s_at, pr["s_m"], pr["h"][0])
    frames = sorted((ROOT / "runs" / a.run_id / "frames").glob("frame_*.npz"))[::a.every]
    hmax = 1.0
    for fp in frames[::max(1, len(frames) // 12)]:
        h = sample(np.load(fp), nb, z.size)[0]; hmax = max(hmax, float(h.max()))
    out = ROOT / "figures" / "animations" / a.run_id / "profile"; out.mkdir(parents=True, exist_ok=True)
    ctx = {"chain": chain, "bed": bed, "nb": nb, "n_cells": z.size, "h_base": hb, "stations": stations, "join_km": join_km,
           "exag": a.exag, "smooth": a.smooth, "zmin": float(bed.min()) - 100, "zmax": float(bed.max()) + 300, "hmax": a.hmax if a.hmax > 0 else hmax * 1.05,
           "label": a.label, "dpi": a.dpi, "out": out, "r1d": r1d}
    if a.paper_frames:
        ctx["clean"] = True; ctx["out"] = out.parent / "paper"; ctx["out"].mkdir(parents=True, exist_ok=True); _init(ctx)
        for i in [int(s) for s in a.paper_frames.split(",")]:
            render_one((i, frames[i])); print("paper panel", i, flush=True)
        return
    with Pool(a.workers, initializer=_init, initargs=(ctx,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(render_one, list(enumerate(frames)), chunksize=4)):
            if i % 40 == 0:
                print(f"profile frames {i+1}/{len(frames)}", flush=True)
    video = out.parent / f"{a.run_id}_profile_1d.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(a.states_per_second), "-i",
           str(out / "p_%05d.png"), "-r", str(a.fps), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(video)]
    subprocess.run(cmd, check=True)
    print(video)


if __name__ == "__main__":
    main()
