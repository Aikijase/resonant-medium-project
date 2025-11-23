#!/usr/bin/env python3
# run_snbao_resonant.py
#
# BAO runner for wide-form tables: columns like
#   z, DM_over_r_d, DH_over_r_d, [DV_over_r_d]
#
# It:
# - builds the model vector m in the SAME stacking you request (pair or block)
# - profiles a single amplitude beta (absorbs r_d etc.)
# - (optionally) applies the viscous-DM hook to growth-like kinds (FS8/BETA) — OFF for DM/DH/DV by default
# - writes a per-bin dump and meta JSON
#
# CLI example:
#   python3 run_snbao_resonant.py \
#     --bao-csv bao_measurements.csv \
#     --cov-csv bao_covariance.csv \
#     --stack-style block \
#     --Om0 0.3 --h 0.7 \
#     --out-root outputs/bao_fit
#
# If your covariance was built for interleaved pairs (DM(z1),DH(z1),DM(z2),DH(z2),...),
# use --stack-style pair. If it was built for blocks (all DM, then all DH, then DV),
# use --stack-style block (default).

import os, json, math, csv
from typing import List, Tuple
import numpy as np
import pandas as pd

# Optional fast Cholesky
try:
    from scipy.linalg import cho_factor, cho_solve
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Optional viscous BAO hook (safe; does nothing unless enabled + configured)
try:
    from tools.bao_visc_hook import maybe_scale_model_vector as _maybe_scale_model_vector
    _HAS_BAO_HOOK = True
except Exception:
    _HAS_BAO_HOOK = False

# ---------------- cosmology (flat LCDM) ----------------

C_KM_S = 299792.458  # km/s

def E_of_z(z: float, Om0: float) -> float:
    return math.sqrt(Om0*(1+z)**3 + (1-Om0))

def H_of_z(z: float, h: float, Om0: float) -> float:
    return 100.0*h * E_of_z(z, Om0)  # km/s/Mpc

def chi_comoving(z: float, h: float, Om0: float, nsteps: int = 2048) -> float:
    H0 = 100.0*h
    if z == 0.0:
        return 0.0
    z_grid = np.linspace(0.0, z, nsteps)
    Ez = np.sqrt(Om0*(1.0+z_grid)**3 + (1.0-Om0))
    integral = np.trapz(1.0/Ez, z_grid)
    return (C_KM_S / H0) * integral  # Mpc

def LCDM_distances(z: np.ndarray, Om0: float, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    DM = np.zeros_like(z, float)
    DH = np.zeros_like(z, float)
    DV = np.zeros_like(z, float)
    for i, zz in enumerate(z):
        DM[i] = chi_comoving(float(zz), h, Om0)
        Hz = H_of_z(float(zz), h, Om0)
        DH[i] = C_KM_S / Hz
        DV[i] = (DM[i]**2 * (C_KM_S*zz/Hz))**(1.0/3.0)
    return DM, DH, DV

# ---------------- IO helpers ----------------

def _find_col(df: pd.DataFrame, cands: List[str]) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for k in cands:
        if k in low:
            return low[k]
    return None

def read_bao_wide(path: str) -> dict:
    """
    Read wide-form BAO table. Returns a dict with keys:
      'z', and any of {'DM','DH','DV'} present (values are np arrays).
    Accepts common header variants.
    """
    df = pd.read_csv(path)
    zcol = _find_col(df, ["z","z_eff","redshift"])
    if not zcol:
        raise ValueError(f"Could not find a redshift column in {list(df.columns)}")

    # Accept a few header spellings
    DMcol = _find_col(df, ["dm_over_r_d","dm_over_rd","dm_over_rdrag","dm"])
    DHcol = _find_col(df, ["dh_over_r_d","dh_over_rd","dh_over_rdrag","dh","hz_over_rd","hz_over_r_d"])
    DVcol = _find_col(df, ["dv_over_r_d","dv_over_rd","dv_over_rdrag","dv"])

    z = pd.to_numeric(df[zcol], errors="coerce").to_numpy()
    out = {"z": z}

    if DMcol is not None: out["DM"] = pd.to_numeric(df[DMcol], errors="coerce").to_numpy()
    if DHcol is not None: out["DH"] = pd.to_numeric(df[DHcol], errors="coerce").to_numpy()
    if DVcol is not None: out["DV"] = pd.to_numeric(df[DVcol], errors="coerce").to_numpy()

    if np.any(~np.isfinite(z)):
        raise ValueError("Non-finite redshifts found.")
    if not any(k in out for k in ("DM","DH","DV")):
        raise ValueError(f"No BAO observable columns recognized in {list(df.columns)}")

    # Ensure equal lengths where present
    n = len(z)
    for key in ("DM","DH","DV"):
        if key in out and len(out[key]) != n:
            raise ValueError(f"Column length mismatch for {key}: {len(out[key])} vs z {n}")
    return out

def read_cov_csv(path: str) -> np.ndarray:
    C = np.loadtxt(path, delimiter=",")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Covariance must be square; got {C.shape}")
    return np.asarray(C, float)

# ---------------- vector builders ----------------

def build_vectors_pairwise(z: np.ndarray, DM: np.ndarray | None, DH: np.ndarray | None,
                           DV: np.ndarray | None) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Order: for each row i, append available observables in DM,DH,DV order.
    Example for DM & DH present: [DM(z1),DH(z1), DM(z2),DH(z2), ...]
    """
    y_list, kinds, z_list = [], [], []
    for i in range(len(z)):
        if DM is not None and np.isfinite(DM[i]):
            y_list.append(float(DM[i])); kinds.append("DM"); z_list.append(float(z[i]))
        if DH is not None and np.isfinite(DH[i]):
            y_list.append(float(DH[i])); kinds.append("DH"); z_list.append(float(z[i]))
        if DV is not None and np.isfinite(DV[i]):
            y_list.append(float(DV[i])); kinds.append("DV"); z_list.append(float(z[i]))
    return np.array(z_list), kinds, np.array(y_list)

def build_vectors_block(z: np.ndarray, DM: np.ndarray | None, DH: np.ndarray | None,
                        DV: np.ndarray | None) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Order: all DM (in ascending z), then all DH, then all DV.
    """
    y_list, kinds, z_list = [], [], []
    if DM is not None:
        for i in range(len(z)):
            if np.isfinite(DM[i]): y_list.append(float(DM[i])); kinds.append("DM"); z_list.append(float(z[i]))
    if DH is not None:
        for i in range(len(z)):
            if np.isfinite(DH[i]): y_list.append(float(DH[i])); kinds.append("DH"); z_list.append(float(z[i]))
    if DV is not None:
        for i in range(len(z)):
            if np.isfinite(DV[i]): y_list.append(float(DV[i])); kinds.append("DV"); z_list.append(float(z[i]))
    return np.array(z_list), kinds, np.array(y_list)

# ---------------- fitting ----------------

def profile_beta_and_dump(z_vec: np.ndarray, kinds: List[str], y_vec: np.ndarray,
                          Cbao: np.ndarray, Om0: float, h: float, out_root: str,
                          stack_style: str, src_csv: str, cov_csv: str) -> None:

    # Build theoretical distances on the per-entry z vector
    DM_all, DH_all, DV_all = LCDM_distances(z_vec, Om0, h)

    # Assemble model vector in the same order as kinds
    m_list = []
    iDM = iDH = iDV = 0
    for k in kinds:
        if k == "DM": m_list.append(DM_all[iDM]); iDM += 1
        elif k == "DH": m_list.append(DH_all[iDH]); iDH += 1
        elif k == "DV": m_list.append(DV_all[iDV]); iDV += 1
        elif k in ("FS8","BETA"):
            m_list.append(1.0)
        else:
            raise ValueError(f"Unknown kind '{k}' in build loop.")
    m = np.array(m_list, float)
    y = np.array(y_vec, float)

    # Optional viscous hook (safe; off for BAO kinds by default)
    if _HAS_BAO_HOOK:
        z_for_each_entry = list(z_vec)
        m, _ = _maybe_scale_model_vector(m, kinds, z_for_each_entry)

    # Jittered covariance for stability
    if Cbao.shape[0] != len(y):
        raise ValueError(f"Covariance size {Cbao.shape[0]} != data length {len(y)}. "
                         f"Try changing --stack-style (current: {stack_style}).")
    eps = 1e-12 * float(np.median(np.diag(Cbao)))
    Cf = np.asarray(Cbao, float) + eps * np.eye(Cbao.shape[0])

    # GLS β = (m^T C^{-1} y)/(m^T C^{-1} m)
    if _HAS_SCIPY:
        cf = cho_factor(Cf, lower=True, check_finite=False)
        Ci = lambda v: cho_solve(cf, v, check_finite=False)
    else:
        Ci = lambda v: np.linalg.solve(Cf, v)

    denom = float(m @ Ci(m))
    if abs(denom) < 1e-300: denom = 1e-300
    beta = float((m @ Ci(y)) / denom)
    y_model = beta * m
    resid = y - y_model
    std = np.sqrt(np.clip(np.diag(Cf), 1e-300, None))
    std_resid = resid / std

    # Outputs
    out_dir = os.path.dirname(out_root)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    tag = os.environ.get("DUMP_VECTOR_TAG", "LCDM")

    dump_csv = f"{out_root}_{tag}.csv"
    pd.DataFrame({
        "z": z_vec,
        "kind": np.array(kinds, str),
        "y_data": y,
        "y_model": y_model,
        "residual": resid,
        "std_resid": std_resid
    }).to_csv(dump_csv, index=False)

    meta = {
        "Om0": Om0, "h": h, "beta": beta,
        "n": int(len(y)),
        "stack_style": stack_style,
        "bao_csv": os.path.abspath(src_csv),
        "cov_csv": os.path.abspath(cov_csv),
        "visc_hook_available": bool(_HAS_BAO_HOOK),
        "env": {
            "DM_GRID": os.environ.get("DM_GRID"),
            "DM_KEFF": os.environ.get("DM_KEFF"),
            "BAO_SCALE": os.environ.get("BAO_SCALE"),
            "BAO_SCALE_KINDS": os.environ.get("BAO_SCALE_KINDS"),
            "BAO_VISCREPORT": os.environ.get("BAO_VISCREPORT"),
        }
    }
    with open(f"{out_root}_{tag}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[bao] β (profiled) = {beta:.6g}")
    print("[bao] wrote", dump_csv)
    print("[bao] wrote", f"{out_root}_{tag}_meta.json")

# ---------------- main ----------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="BAO wide-form runner (supports DM/DH/DV, pair/block stacking).")
    ap.add_argument("--bao-csv", type=str, default="bao_measurements.csv", help="Wide-form BAO CSV.")
    ap.add_argument("--cov-csv", type=str, default="bao_covariance.csv", help="Covariance CSV (square).")
    ap.add_argument("--stack-style", choices=["pair","block"], default="block",
                    help="Order of the data vector. 'pair' ≈ [DM(z1),DH(z1),...]; 'block' ≈ [DM(..), DH(..), DV(..)].")
    ap.add_argument("--Om0", type=float, default=0.3, help="Ω_m,0 (flat LCDM).")
    ap.add_argument("--h",   type=float, default=0.7, help="little h.")
    ap.add_argument("--out-root", type=str, default="outputs/bao_fit", help="Output root (no extension).")
    args = ap.parse_args()

    tbl = read_bao_wide(args.bao_csv)
    z = np.array(tbl["z"], float)
    DM = tbl.get("DM", None)
    DH = tbl.get("DH", None)
    DV = tbl.get("DV", None)

    if args.stack_style == "pair":
        z_vec, kinds, y_vec = build_vectors_pairwise(z, DM, DH, DV)
    else:
        z_vec, kinds, y_vec = build_vectors_block(z, DM, DH, DV)

    C = read_cov_csv(args.cov_csv)
    if C.shape[0] != len(y_vec):
        print(f"[warn] Covariance size {C.shape[0]} != data length {len(y_vec)} for stack '{args.stack_style}'.")
        print("       If this is unexpected, try the other --stack-style (pair ↔ block).")

    profile_beta_and_dump(z_vec, kinds, y_vec, C, args.Om0, args.h,
                          args.out_root, args.stack_style, args.bao_csv, args.cov_csv)

if __name__ == "__main__":
    main()
