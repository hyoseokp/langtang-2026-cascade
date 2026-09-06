"""CPU sanity checks for solver_twophase: lake at rest, mass conservation, drag locking, single-phase equivalence."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from solver_twophase import TwoPhaseSWE
from solver_sparse import SparseSWE

torch.manual_seed(0)
rows, cols, cell = 40, 60, 30.0
yy, xx = np.mgrid[0:rows, 0:cols]
z = 20.0 * np.sin(xx / 5.0) * np.cos(yy / 4.0) + 0.02 * xx * cell  # rough sloping bed
z[:2, :] = z[-2:, :] = 500.0; z[:, :2] = z[:, -2:] = 500.0  # closed basin (edge cells are zeroed by the solver)
active = np.ones((rows, cols), bool)


def make(cls, **kw):
    return cls(z.astype(np.float32), active, cell, device="cpu", mu_s=kw.pop("mu_s", 0.4), n_w=0.03, n_d=0.03, **kw)


# 1. lake at rest with both phases present
m = make(TwoPhaseSWE, u_t=1.0)
eta = 60.0
h = np.clip(eta - z, 0, None).astype(np.float32)
m.hs = torch.as_tensor(0.4 * h.ravel()); m.hf = torch.as_tensor(0.6 * h.ravel())
for _ in range(200):
    m.step(m.max_dt())
us, vs, uf, vf = m.phase_velocities()
print(f"lake at rest: max |u_s| {float(us.abs().max()):.2e} |u_f| {float(uf.abs().max()):.2e} m/s, surface dev {float(((m.z + m.h) - eta)[m.h > 0.5].abs().max()):.2e} m")

# 2. dam break: mass conservation per phase, phases lock with strong drag
m = make(TwoPhaseSWE, u_t=1.0)
hs0 = np.zeros((rows, cols), np.float32); hs0[15:25, 5:15] = 20.0
hf0 = np.zeros((rows, cols), np.float32); hf0[15:25, 5:15] = 5.0; hf0[:, 30:] = 1.0
m.hs = torch.as_tensor(hs0.ravel()); m.hf = torch.as_tensor(hf0.ravel())
Vs0, Vf0 = float(m.hs.sum()), float(m.hf.sum())
t = 0.0
while t < 120:
    dt = m.max_dt(); m.step(dt); t += dt
us, vs, uf, vf = m.phase_velocities(); both = (m.hs > 0.05) & (m.hf > 0.05)
print(f"dam break U_T=1: mass err solid {float(m.hs.sum())/Vs0-1:.2e} fluid {float(m.hf.sum())/Vf0-1:.2e}, "
      f"max |u_f-u_s| where both wet {float((uf-us)[both].abs().max()):.3f} m/s, max speed {float(torch.sqrt(us*us+vs*vs).max()):.1f}")

# 3. weak drag: phases separate
m = make(TwoPhaseSWE, u_t=50.0)
m.hs = torch.as_tensor(hs0.ravel()); m.hf = torch.as_tensor(hf0.ravel())
t = 0.0
while t < 120:
    dt = m.max_dt(); m.step(dt); t += dt
us, vs, uf, vf = m.phase_velocities(); both = (m.hs > 0.05) & (m.hf > 0.05)
print(f"dam break U_T=50: max |u_f-u_s| {float((uf-us)[both].abs().max()):.3f} m/s; solid front col "
      f"{int(np.nonzero(m.full(m.hs).max(0) > 0.1)[0].max())} fluid front col {int(np.nonzero(m.full(m.hf).max(0) > 1.2)[0].max())}")

# 4. equivalence with single-phase when mu=0, strong drag, fluid only vs mixture
ms = make(SparseSWE, mu_s=0.0)
ms.h = torch.as_tensor((hs0 + hf0).ravel()); ms.hc = torch.as_tensor(hs0.ravel())
m2 = make(TwoPhaseSWE, mu_s=0.0, u_t=0.1)
m2.hs = torch.as_tensor(hs0.ravel()); m2.hf = torch.as_tensor(hf0.ravel())
t = 0.0
while t < 60:
    dt = min(ms.max_dt(), m2.max_dt()); ms.step(dt); m2.step(dt); t += dt
d = (ms.h - m2.h).abs()
print(f"single vs two-phase (mu=0, locked): max |dh| {float(d.max()):.3f} m, rms {float(torch.sqrt((d*d).mean())):.4f} m, hmax {float(ms.h.max()):.2f}/{float(m2.h.max()):.2f}")
