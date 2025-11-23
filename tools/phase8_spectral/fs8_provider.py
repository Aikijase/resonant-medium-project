#!/usr/bin/env python3
"""
fs8_provider.py — REAL provider via growth.solve_growth

Uses your repo's growth solver:
    growth.solve_growth(p, zmax=..., npts=..., sigma8_0=...)
which returns a dict with keys:
    z (ascending), D (normalized so D[z=0]=1), and fs8 (or f*sigma8)

We build two adapters:
- LCDM:    tau=0, kappa=0 (no memory), other params as given
- MODEL:   tau, kappa from user (memory on)

We then interpolate fs8(z) onto the requested a-grid via z = 1/a - 1.

If growth.solve_growth changes its API, tweak _call_growth() below.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

# --- import your real solver ---
from growth import solve_growth  # <-- from ./growth.py in your repo


@dataclass
class Params:
    mu0: float
    nu0: float
    tau: float
    kappa: float
    sigma8: float
    Om: float   # Om0 today


# ---------- helpers ----------

def _validate_1d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} must be finite 1D array, got shape {arr.shape}")
    return arr

def _interp_fs8_at_a(a: np.ndarray, z: np.ndarray, fs8_z: np.ndarray) -> np.ndarray:
    """Interpolate fs8(z) onto a-grid. z is ascending; a typically ascending too."""
    A = _validate_1d("a", a)
    Z = _validate_1d("z", z)
    F = _validate_1d("fs8(z)", fs8_z)

    # Convert requested a to z; clamp to solver range
    z_req = 1.0 / np.clip(A, 1e-12, None) - 1.0
    z_min, z_max = float(Z.min()), float(Z.max())
    z_req = np.clip(z_req, z_min, z_max)

    # Z is ascending; np.interp expects ascending x
    fs8_a = np.interp(z_req, Z, F)
    return np.clip(fs8_a, 1e-12, None)

def _call_growth(pdict: Dict[str, Any], *, zmax: float, npts: int, sigma8_0: float):
    """Call your growth.solve_growth and return (z, fs8) as 1D arrays."""
    out = solve_growth(pdict, zmax=zmax, npts=npts, sigma8_0=sigma8_0)
    # Defensive extraction: accept 'fs8' OR 'f*sigma8' keys.
    z   = np.asarray(out.get("z"), dtype=float)
    fs8 = out.get("fs8", None)
    if fs8 is None:
        fs8 = out.get("f*sigma8", None)
    if fs8 is None:
        raise RuntimeError("solve_growth did not return 'fs8' or 'f*sigma8'. "
                           "Check growth.py output keys.")
    fs8 = np.asarray(fs8, dtype=float)
    # Sanity
    if z.ndim != 1 or fs8.ndim != 1 or len(z) != len(fs8):
        raise RuntimeError(f"solve_growth returned mismatched shapes: z{z.shape}, fs8{fs8.shape}")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(fs8))):
        raise RuntimeError("solve_growth returned non-finite values.")
    # Ensure z ascending for interp
    if not np.all(np.diff(z) >= 0):
        idx = np.argsort(z)
        z, fs8 = z[idx], fs8[idx]
    return z, fs8


# ---------- public API ----------

def get_fs8_series(a: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (fs8_LCDM(a), fs8_MODEL(a)) using your real growth solver.
    """
    A = _validate_1d("a", np.asarray(a, dtype=float))

    # Build parameter dicts expected by growth.solve_growth
    # Known keys from your growth.py path (via KeyError): A, f, phi, gamma.
    # We pass sensible defaults; adjust if your physical model needs specific values.
    def _mk_pdict(tau_val: float, kappa_val: float) -> dict:
        d = dict(
            # cosmology / medium knobs we already use
            Om0=p.Om, mu0=p.mu0, nu0=p.nu0, tau=tau_val, kappa=kappa_val,
            # provide defaults required by growth.py's H_of_z():
            A=1.0,          # amplitude-like (placeholder; set to your model's value if needed)
            f=0.0,          # extra coupling term (placeholder)
            phi=0.0,        # potential-like phase (placeholder)
            gamma=0.55,     # growth index ~0.55 for GR+ΛCDM; adjust for your model if needed
        )
        return d

    # LCDM: zero memory knobs
    p_lcdm = _mk_pdict(tau_val=0.0, kappa_val=0.0)

    # MODEL: use requested memory knobs
    p_model = _mk_pdict(tau_val=p.tau, kappa_val=p.kappa)

    # Decide how far in z we need: a in [amin, 1] -> z in [0, zmax]
    a_min = float(np.min(A))
    zmax = max(0.0, 1.0 / max(a_min, 1e-8) - 1.0)

    # Resolution: npts ~ length of a but keep it modestly dense
    npts = max(200, min(2000, 2 * len(A)))

    # Call solver
    zL, fs8L = _call_growth(p_lcdm, zmax=zmax, npts=npts, sigma8_0=p.sigma8)
    zM, fs8M = _call_growth(p_model, zmax=zmax, npts=npts, sigma8_0=p.sigma8)

    # Interpolate onto requested a-grid
    fL = _interp_fs8_at_a(A, zL, fs8L)
    fM = _interp_fs8_at_a(A, zM, fs8M)

    return fL, fM
