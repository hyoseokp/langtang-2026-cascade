"""Render stored SWE frames into top-view and oblique MP4 videos. Reads saved data only."""

from __future__ import annotations

import os

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
NPT = timezone(timedelta(hours=5, minutes=45))
ONSET_UTC = datetime(2026, 8, 26, 2, 52, 10, tzinfo=timezone.utc)
DEBRIS_RGB = np.array([0.95, 0.45, 0.05])
WATER_RGB = np.array([0.05, 0.25, 0.90])
BASE_RGB = np.array([0.55, 0.65, 0.75])

_CTX: dict = {}


def load_inputs(tag: str):
    inp = np.load(ROOT / "inputs" / f"{tag}.npz")
    meta = json.loads((ROOT / "inputs" / f"{tag}.json").read_text("utf-8"))
    return inp, meta


def hillshade(z: np.ndarray, cell: float) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=45)
    lo, hi = np.nanpercentile(z, [1, 99.5])
    return ls.shade(z, cmap=plt.get_cmap("gray"), vert_exag=1.0, dx=cell, dy=cell,
                    vmin=lo, vmax=hi, blend_mode="soft")[..., :3]


def overlay_rgba(shape, frame, h_base, *, depth_scale: float = 1.5):
    """Colour by debris fraction, opacity by excess depth above the pre-event river."""
    h = np.zeros(shape, np.float32).ravel()
    c = np.zeros(shape, np.float32).ravel()
    h[frame["idx"]] = frame["h"].astype(np.float32)
    c[frame["idx"]] = frame["c"].astype(np.float32)
    h = h.reshape(shape); c = c.reshape(shape)
    excess = np.maximum(h - h_base, 0.0)
    debris_depth = h * c
    w = np.clip(c / 0.35, 0.0, 1.0)[..., None]  # >= 35 % solids reads as debris; diluted flood water stays blue
    rgb = WATER_RGB[None, None, :] * (1 - w) + DEBRIS_RGB[None, None, :] * w
    signal = np.maximum(excess, debris_depth)
    alpha = np.where(signal > 0.2, 0.25 + 0.75 * (1 - np.exp(-signal / depth_scale)), 0.0)
    base_alpha = np.where((h_base > 0.3) & (alpha == 0), 0.55, 0.0)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.float32)
    base = np.concatenate([np.broadcast_to(BASE_RGB, shape + (3,)), base_alpha[..., None]], axis=-1).astype(np.float32)
    k = _CTX.get("dilate", 0)
    if k:
        from scipy import ndimage
        for arr in (rgba, base):
            m = arr[..., 3] > 0
            if not m.any():
                continue
            grown = ndimage.binary_dilation(m, iterations=k)
            _, (ri, ci) = ndimage.distance_transform_edt(~m, return_indices=True)
            fill = grown & ~m
            arr[fill] = arr[ri[fill], ci[fill]]
    return rgba, base, h, c


def _init_worker(ctx):
    _CTX.update(ctx)


def station_marker(ax, stations, extent, three_d=False, z_fn=None):
    x0, x1, y0, y1 = extent
    for key, s in stations.items():
        if not (x0 <= s["x"] <= x1 and y0 <= s["y"] <= y1):
            continue
        mk = "*" if key == "usgs_main_onset" else ("o" if "cctv" in key else "s")
        if three_d:
            ax.scatter(s["x"], s["y"], z_fn(s["x"], s["y"]) + 150, marker=mk, s=40, color="white",
                       edgecolor="black", depthshade=False)
        else:
            ax.scatter(s["x"], s["y"], marker=mk, s=45, facecolor="white", edgecolor="black", linewidth=0.8, zorder=9)
            ax.annotate(s["label"], (s["x"], s["y"]), xytext=(6, 5), textcoords="offset points", fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1}, zorder=10)


def render_one(args):
    frame_no, frame_path = args
    ctx = _CTX
    fr = np.load(frame_path)
    t = float(fr["time_s"])
    shape = ctx["z"].shape
    overlays = {}
    clock = (ONSET_UTC + timedelta(seconds=t)).astimezone(NPT)
    row = ctx["series"].get(round(t))
    front_txt = ""
    if row:
        front_txt = (f"debris front (route) {float(row['debris_front_route_km']):.1f} km | "
                     f"flood front {float(row['flood_front_route_km']):.1f} km\n"
                     f"mobile volume {float(row['total_volume_m3'])/1e6:.1f} Mm3, debris {float(row['debris_volume_m3'])/1e6:.1f} Mm3")
    title = (f"{ctx['label']}\nt = {t:.0f} s ({t/3600:.2f} h) | {clock:%H:%M:%S} NPT, 26 Aug 2026")
    outputs = []
    for view in ctx["views"]:
        k_dil = view.get("dilate", ctx.get("dilate", 0))
        if k_dil not in overlays:
            _CTX["dilate"] = k_dil; overlays[k_dil] = overlay_rgba(shape, fr, ctx["h_base"])
        rgba, base, h, c = overlays[k_dil]
        ext = view["extent"]
        x0, x1, y0, y1 = ext
        r0, r1, c0, c1 = view["rc"]
        w, hgt = view["figsize"]
        fig, ax = plt.subplots(figsize=(w, hgt), dpi=ctx["dpi"])
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.9)
        full = ctx["extent"]
        ax.imshow(ctx["shade"][r0:r1, c0:c1], extent=ext, origin="upper", interpolation="nearest")
        ax.imshow(base[r0:r1, c0:c1], extent=ext, origin="upper", interpolation="nearest")
        ax.imshow(rgba[r0:r1, c0:c1], extent=ext, origin="upper", interpolation="nearest")
        if ctx.get("obs") is not None:
            xs = np.linspace(x0, x1, c1 - c0, endpoint=False) + 30
            ys = np.linspace(y1, y0, r1 - r0, endpoint=False) - 30
            ax.contour(xs, ys, ctx["obs"][r0:r1, c0:c1].astype(float), levels=[0.5], colors=["#00e5ff"],
                       linewidths=0.6)
        station_marker(ax, ctx["stations"], ext)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UTM 45N easting (m)"); ax.set_ylabel("UTM 45N northing (m)")
        ax.tick_params(labelsize=8)
        handles = [Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=col, markeredgecolor="none",
                          markersize=8, label=lab) for lab, col in
                   (("debris flow (solids >= 35 %)", DEBRIS_RGB), ("flood water (solids -> 0)", WATER_RGB),
                    ("pre-event river", BASE_RGB))]
        handles.append(Line2D([0], [0], color="#00e5ff", linewidth=1.2, label="UNOSAT mapped affected surface"))
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)
        ax.text(0.99, 0.01, front_txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        name_txt = view["name"] + (f" | wet cells widened by {k_dil} cells for display" if k_dil else "")
        ax.text(0.01, 0.99, name_txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        out = ctx["out"] / f"{view['key']}_top" / f"f_{frame_no:05d}.png"
        fig.savefig(out); plt.close(fig); outputs.append(out)
        if view.get("oblique"):
            st = view["stride"]
            zz = ctx["z"][r0:r1:st, c0:c1:st]
            hh = h[r0:r1:st, c0:c1:st]
            surf = zz + hh * ctx["exag"]
            xx = ctx["xx"][r0:r1:st, c0:c1:st]; yy = ctx["yy"][r0:r1:st, c0:c1:st]
            a = rgba[r0:r1:st, c0:c1:st, 3:4]
            b = base[r0:r1:st, c0:c1:st, 3:4]
            face = ctx["shade"][r0:r1:st, c0:c1:st] * (1 - b) + BASE_RGB * b
            face = face * (1 - a) + rgba[r0:r1:st, c0:c1:st, :3] * a
            fig = plt.figure(figsize=(w, hgt), dpi=ctx["dpi"])
            ax3 = fig.add_subplot(111, projection="3d")
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.92)
            ax3.plot_surface(xx, yy, surf, facecolors=face, rstride=1, cstride=1, linewidth=0,
                             antialiased=False, shade=False)
            zmin, zmax = view["zlim"]
            ax3.set_zlim(zmin, zmax)
            ax3.set_xlim(x0, x1); ax3.set_ylim(y0, y1)
            ax3.set_box_aspect((x1 - x0, y1 - y0, (zmax - zmin) * view["zscale"]))
            ax3.view_init(elev=view["elev"], azim=view["azim"])
            ax3.set_axis_off()
            ax3.set_title(title, fontsize=11)
            ax3.text2D(0.01, 0.02, f"{view['name']} | oblique view, flow thickness x{ctx['exag']:g} for display, "
                       f"terrain vertical scale x{view['zscale']:g}; mesh stride {st} (statistics on the 60 m grid)\n{front_txt}",
                       transform=ax3.transAxes, fontsize=8)
            out = ctx["out"] / f"{view['key']}_oblique" / f"f_{frame_no:05d}.png"
            fig.savefig(out); plt.close(fig); outputs.append(out)
    return frame_no


def encode(pattern: Path, output: Path, fps_states: float, fps_out: int) -> None:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps_states),
           "-i", str(pattern), "-r", str(fps_out), "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--base-run", default="spinup60_v1")
    p.add_argument("--tag", default="corridor60")
    p.add_argument("--label", default="INDEPENDENT MIXTURE-SWE RECONSTRUCTION (diagnostic)")
    p.add_argument("--states-per-second", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--exag", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--display-dilate", type=int, default=0, help="display-only widening of wet cells (cells)")
    p.add_argument("--views", default="", help="comma-separated view keys to render (default all)")
    p.add_argument("--stride", type=int, default=0, help="override the oblique mesh stride")
    a = p.parse_args()

    inp, meta = load_inputs(a.tag)
    z = inp["z"].astype(np.float32)          # carved simulation bed (consistent with stored depths)
    cell = meta["cell_m"]
    left, right, bottom, top = meta["extent_utm45n"]
    rows, cols = z.shape
    shade = hillshade(z, cell)
    xs = left + (np.arange(cols) + 0.5) * cell
    ys = top - (np.arange(rows) + 0.5) * cell
    xx, yy = np.meshgrid(xs, ys)
    base_state = np.load(ROOT / "runs" / a.base_run / "final_state.npz")
    h_base = base_state["h"].astype(np.float32)
    run_dir = ROOT / "runs" / a.run_id
    frames = sorted((run_dir / "frames").glob("frame_*.npz"))[::a.every]
    if a.max_frames:
        frames = frames[:a.max_frames]
    series = {}
    with (run_dir / "series.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series[round(float(row["time_s"]))] = row
    stations = meta["stations"]

    def rc(x, y):
        return int((top - y) // cell), int((x - left) // cell)

    # upper view: ~45 km window over the Langtang gorge (source, Gyirong, Syabrubesi, Betrawati reach)
    ux0, ux1, uy0, uy1 = 316000.0, right, 3092000.0, top  # to the grid's north and east edges so the source area is complete
    ur0, uc0 = rc(ux0, uy1); ur1, uc1 = rc(ux1, uy0)
    ur0, uc0 = max(ur0, 0), max(uc0, 0); ur1, uc1 = min(ur1, rows), min(uc1, cols)
    uext = (left + uc0 * cell, left + uc1 * cell, top - ur1 * cell, top - ur0 * cell)
    zu = z[ur0:ur1, uc0:uc1]
    if a.tag.startswith("corridor"):
        views = [
            {"key": "full", "name": "Full corridor: source to Devghat", "extent": (left, right, bottom, top),
             "rc": (0, rows, 0, cols), "figsize": (16, 10.6), "oblique": True, "stride": 6, "dilate": 4,
             "zlim": (float(z.min()), float(z.max()) + 200), "zscale": 3.0, "elev": 45, "azim": 235},
            {"key": "upper", "name": "Langtang gorge, ~45 km window: source to below Syabrubesi", "extent": uext,
             "rc": (ur0, ur1, uc0, uc1), "figsize": (12, 13), "oblique": True, "stride": 2, "dilate": 2,
             "zlim": (float(zu.min()), float(zu.max()) + 200), "zscale": 1.0, "elev": 50, "azim": 235},
        ]
    else:
        views = [
            {"key": "upper30", "name": f"Upper corridor at {cell:g} m: source to Syabrubesi", "extent": (left, right, bottom, top),
             "rc": (0, rows, 0, cols), "figsize": (12, 12.5), "oblique": True, "stride": 3,
             "zlim": (float(z.min()), float(z.max()) + 200), "zscale": 1.0, "elev": 50, "azim": 235},
        ]
    if a.views:
        views = [v for v in views if v["key"] in a.views.split(",")]
    if a.stride:
        for v in views:
            v["stride"] = a.stride
    out = ROOT / "figures" / "animations" / a.run_id
    for v in views:
        (out / f"{v['key']}_top").mkdir(parents=True, exist_ok=True)
        (out / f"{v['key']}_oblique").mkdir(parents=True, exist_ok=True)
    obs_path = ROOT / "inputs" / f"{a.tag}_observed_footprint.npz"
    obs = np.load(obs_path)["unosat_affected"] if obs_path.exists() else None
    if obs is not None and obs.shape != z.shape:
        obs = None
    ctx = {"z": z, "shade": shade, "xx": xx, "yy": yy, "h_base": h_base, "extent": (left, right, bottom, top), "obs": obs,
           "stations": stations, "views": views, "out": out, "dpi": a.dpi, "label": a.label, "exag": a.exag,
           "series": series, "dilate": a.display_dilate}
    jobs = list(enumerate(frames))
    with Pool(a.workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(render_one, jobs, chunksize=2)):
            if i % 20 == 0:
                print(f"rendered {i + 1}/{len(jobs)}", flush=True)
    videos = {}
    for v in views:
        for kind in ("top", "oblique"):
            vid = out / f"{a.run_id}_{v['key']}_{kind}.mp4"
            encode(out / f"{v['key']}_{kind}" / "f_%05d.png", vid, a.states_per_second, a.fps)
            videos[vid.name] = hashlib.sha256(vid.read_bytes()).hexdigest()
            print(vid)
    prov = {"run_id": a.run_id, "base_run": a.base_run, "frames": len(frames), "frame_step": a.every,
            "states_per_second": a.states_per_second, "fps": a.fps, "flow_thickness_exaggeration": a.exag,
            "label": a.label, "views": [{k: (list(v[k]) if isinstance(v[k], tuple) else v[k]) for k in v} for v in views],
            "sha256": videos}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
