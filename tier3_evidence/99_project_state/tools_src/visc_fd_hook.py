# tools/visc_fd_hook.py
# Robust helper to read a viscous-DM growth grid and compute (fD)_visc/(fD)_LCDM.
# Driven by env vars DM_GRID (path to *_growth_grid.csv) and DM_KEFF (h/Mpc).

import os, csv, numpy as np
from typing import Tuple

_DM_GRID = os.environ.get("DM_GRID")                  # path to *_growth_grid.csv
_DM_KEFF = float(os.environ.get("DM_KEFF", "0.20"))   # default k_eff [h/Mpc]
_GRID = None  # lazy-loaded cache

_TINY = 1e-300

def _safelog(x):
    return np.log(np.maximum(x, _TINY))

def _read_growth_grid_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = np.array([[float(x) for x in row] for row in r], float)
    a = rows[:, 0]
    D_LCDM = rows[:, 1]
    k_list, Dk = [], []
    for j, h in enumerate(header[2:], start=2):
        if h.startswith("D_visc_k="):
            k = float(h.split("=")[1])
            k_list.append(k)
            Dk.append(rows[:, j])
    return a, D_LCDM, np.array(k_list, float), np.array(Dk, float)  # Dk: (nk, na)

def _ensure_loaded():
    global _GRID
    if _GRID is None and _DM_GRID:
        _GRID = _read_growth_grid_csv(_DM_GRID)
    return _GRID

def _nearest_idx(vec: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(vec - x)))

def _dlnD_dlnA(D_vec: np.ndarray, a_vec: np.ndarray, i: int) -> float:
    i = max(1, min(i, len(a_vec) - 2))
    lnD1 = _safelog(D_vec[i+1]); lnD0 = _safelog(D_vec[i-1])
    lna1 = np.log(a_vec[i+1]);   lna0 = np.log(a_vec[i-1])
    denom = lna1 - lna0
    if not np.isfinite(denom) or denom <= 0:
        # fallback to a wider stencil
        il = max(0, i-2); ir = min(len(a_vec)-1, i+2)
        if il == ir or a_vec[ir] == a_vec[il]:
            return 0.0
        return float((_safelog(D_vec[ir]) - _safelog(D_vec[il])) / (np.log(a_vec[ir]) - np.log(a_vec[il])))
    return float((lnD1 - lnD0) / denom)

def _bilinear_D(Dk: np.ndarray, a_vec: np.ndarray, k_vec: np.ndarray, a_req: float, k_req: float) -> float:
    # indices in a
    ia1 = int(np.searchsorted(a_vec, a_req)); ia0 = max(0, ia1-1); ia1 = min(ia1, len(a_vec)-1)
    a_w = 0.0 if ia0 == ia1 else (a_req - a_vec[ia0]) / (a_vec[ia1] - a_vec[ia0])
    # indices in k
    ik1 = int(np.searchsorted(k_vec, k_req)); ik0 = max(0, ik1-1); ik1 = min(ik1, len(k_vec)-1)
    k_w = 0.0 if ik0 == ik1 else (k_req - k_vec[ik0]) / (k_vec[ik1] - k_vec[ik0])
    # bilinear combine
    D_00 = Dk[ik0, ia0]; D_01 = Dk[ik0, ia1]
    D_10 = Dk[ik1, ia0]; D_11 = Dk[ik1, ia1]
    D_a0 = D_00*(1-a_w) + D_01*a_w
    D_a1 = D_10*(1-a_w) + D_11*a_w
    out = D_a0*(1-k_w) + D_a1*k_w
    return float(out)

def visc_fd_ratio(z: float, k_eff_hMpc: float, grid) -> float:
    """
    Return (fD)_visc/(fD)_LCDM at redshift z and k_eff (h/Mpc).
    Robust to tiny steps and near-grid-edge queries.
    """
    a_vec, D_LCDM, k_vec, Dk = grid
    a = 1.0 / (1.0 + z)
    ia = _nearest_idx(a_vec, a)

    # LCDM part
    D_lcdm = float(np.interp(a, a_vec, D_LCDM))
    f_lcdm = _dlnD_dlnA(D_LCDM, a_vec, ia)
    if not np.isfinite(f_lcdm) or abs(f_lcdm) < 1e-12:
        f_lcdm = 1e-12  # avoid divide-by-zero in the final ratio

    # Viscous part at (a,k)
    D_visc = _bilinear_D(Dk, a_vec, k_vec, a, k_eff_hMpc)

    # Local slope in ln a at fixed k (two-sided finite difference with safety)
    ia_m = max(1, ia-1); ia_p = min(len(a_vec)-2, ia+1)
    Dm = _bilinear_D(Dk, a_vec, k_vec, a_vec[ia_m], k_eff_hMpc)
    Dp = _bilinear_D(Dk, a_vec, k_vec, a_vec[ia_p], k_eff_hMpc)
    num = _safelog(Dp) - _safelog(Dm)
    den = np.log(a_vec[ia_p]) - np.log(a_vec[ia_m])
    if not np.isfinite(den) or den <= 0:
        den = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        f_visc = float(num / den)
    if not np.isfinite(f_visc):
        # fallback: use LCDM slope if the local estimate is ill-conditioned
        f_visc = f_lcdm

    # Final ratio
    D_lcdm = max(D_lcdm, _TINY)
    out = (f_visc * D_visc) / (f_lcdm * D_lcdm)
    return float(out)

def apply_visc_fd_if_available(z: float, fs8_th: float) -> float:
    """Multiply fs8_th by (fD)_visc/(fD)_LCDM if a DM grid is configured via env vars."""
    grid = _ensure_loaded()
    if grid is None:
        return fs8_th
    ratio = visc_fd_ratio(z, _DM_KEFF, grid)
    return fs8_th * float(ratio)
