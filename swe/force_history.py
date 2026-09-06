"""Landslide force history from stored frames: F(t) = -dP/dt, P = sum rho_m h u dx^2 (Ekstrom & Stark 2013 style)."""

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


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--run", default="force30"); p.add_argument("--tag", default="upper30h")
    a = p.parse_args()
    inp = np.load(ROOT / "inputs" / f"{a.tag}.npz"); meta = json.loads((ROOT / "inputs" / f"{a.tag}.json").read_text("utf-8"))
    z = inp["z"].astype(np.float32); cell = meta["cell_m"]; dx2 = cell * cell
    gy, gx = np.gradient(z, cell)  # rows increase southward: gy = dz/d(row)
    zx = gx.ravel(); zy = -gy.ravel()
    rows = []
    for fp in sorted((ROOT / "runs" / a.run / "frames").glob("frame_*.npz")):
        fr = np.load(fp); idx = fr["idx"]; c = fr["c"].astype(np.float32); hu = fr["hu"]; hv = fr["hv"]
        rho = 1000.0 * (1.0 + 1.65 * c)
        px = float((rho * hu).sum()) * dx2; py = float((rho * hv).sum()) * dx2
        pz = float((rho * (hu * zx[idx] + hv * zy[idx])).sum()) * dx2  # along-bed vertical component
        rows.append({"t": float(fr["time_s"]), "Px": px, "Py": py, "Pz": pz,
                     "KE": float((0.5 * rho * (hu * hu + hv * hv) / np.maximum(fr["h"].astype(np.float32), 1e-3)).sum()) * dx2})
    df = pd.DataFrame(rows)
    dt = np.gradient(df.t.to_numpy())
    for k in ("x", "y", "z"):
        df[f"F{k}"] = -np.gradient(df[f"P{k}"].to_numpy(), df.t.to_numpy())  # force on the Earth = -dP/dt of the mass
    df["Fmag"] = np.sqrt(df.Fx ** 2 + df.Fy ** 2 + df.Fz ** 2)
    out = ROOT / "figures" / "paper"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"table_force_history_{a.run}.csv", index=False)
    i = int(df.Fmag.argmax())
    print(f"peak |F| {df.Fmag[i]:.3e} N at t={df.t[i]:.0f} s; peak Fz {df.Fz.abs().max():.3e} N; max momentum {np.sqrt(df.Px**2+df.Py**2).max():.3e} kg m/s; "
          f"peak KE {df.KE.max():.3e} J at t={df.t[int(df.KE.argmax())]:.0f} s")
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)
    ax[0].plot(df.t, df.Fx / 1e11, label="east"); ax[0].plot(df.t, df.Fy / 1e11, label="north"); ax[0].plot(df.t, df.Fz / 1e11, label="up")
    ax[0].plot(df.t, df.Fmag / 1e11, "k--", lw=0.8, label="|F|"); ax[0].set_ylabel("force on Earth (10^11 N)"); ax[0].legend(fontsize=8)
    ax[1].plot(df.t, df.KE / 1e15, color="tab:red"); ax[1].set_ylabel("kinetic energy (10^15 J)"); ax[1].set_xlabel("time after onset (s)")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"fig_force_history_{a.run}.png"); plt.close(fig)


if __name__ == "__main__":
    main()
