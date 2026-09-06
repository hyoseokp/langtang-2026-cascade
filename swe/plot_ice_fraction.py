"""Ice-fraction figure at zero friction (stored tables only): misfit of the upper and lower criteria, melted fraction by 2 h,
and the Devghat excess volume, against the ice fraction of the release."""
import os

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(os.environ.get("LANGTANG_ROOT", str(Path(__file__).resolve().parents[1])))
d = pd.read_csv(ROOT / "figures" / "paper" / "table_ice_mu_grid.csv"); d = d[d.mu == 0].sort_values("ice")
melt2h = []
for ice in d.ice:
    s = pd.read_csv(ROOT / "runs" / f"imu60e_m0_i{ice:g}" / "series.csv"); i = int(np.argmin(np.abs(s.time_s.to_numpy() - 7200)))
    melt2h.append(100 * s.melted_m3.iloc[i] / (35.4e6 * ice))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), dpi=150)
ax[0].plot(d.ice, d.misfit_upper, "o-", color="tab:green", label="upper corridor timing (3)"); ax[0].plot(d.ice, d.misfit_lower, "s-", color="tab:red", label="lower gauges (12)")
ax[0].set_ylabel("misfit"); ax[0].set_ylim(0, 3); ax[0].legend(fontsize=8); ax[0].set_title("fit to the record", fontsize=9)
ax[1].plot(d.ice, melt2h, "o-", color="tab:blue"); ax[1].axhline(95, color="k", ls="--", lw=0.8); ax[1].text(0.62, 96.5, "complete melting", fontsize=7)
ax[1].set_ylabel("release ice melted by 2 h (%)"); ax[1].set_ylim(0, 105); ax[1].set_title("energy bound on melting", fontsize=9)
ax[2].plot(d.ice, d.devghat_excess_Mm3, "o-", color="tab:purple"); ax[2].axhspan(10, 30, color="gold", alpha=0.3); ax[2].set_ylim(0, 35)
ax[2].set_ylabel("Devghat excess volume (10$^6$ m$^3$)"); ax[2].set_title("flood volume at 180 km", fontsize=9)
for a in ax:
    a.set_xlabel("ice fraction of the release (by volume)"); a.set_xlim(0.05, 0.85); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "paper" / "fig_ice_fraction.png"); plt.close(fig)
print(pd.DataFrame({"ice": d.ice, "misfit_upper": d.misfit_upper.round(2), "misfit_lower": d.misfit_lower.round(2), "melt2h_pct": np.round(melt2h, 0), "devghat_excess": d.devghat_excess_Mm3.round(1)}).to_string(index=False))
