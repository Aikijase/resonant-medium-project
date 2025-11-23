#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interacting Dark Sector + BH-reservoir background fitter (SN + BAO)

Key additions vs prior script:
  - BAO covariance handling:
      --bao-cov-kind {cov,prec}  # interpret BAO matrix as covariance or precision
      --bao-cov-scale S          # multiply by S^2 (units/scale fix)
  - SN weighting:
      --sn-weight {auto,diag}
        * auto: use provided SN covariance/precision (--sn-cov, --sn-cov-kind, --sn-scale)
        * diag: use constant sigma (--sn-const-sigma)
  - Analytic marginalization over SN zero-point dμ (on by default; disable with --no-analytic-dmu)

Model summary:
  - Effective fluids (dimensionless densities x = rho/rho_c0):
      x_chi (DM-like), x_phi (DE-like), x_bh (BH reservoir)
  - IDS exchange (windowed):
      q_ids = (epsH * x_chi + xiH * x_phi) * exp[-(z/zR)^2]
      x_chi' = -3 x_chi + q_ids + (1 - fde) * q_bh_out
      x_phi' = -3(1+w_de) x_phi - q_ids + fde * q_bh_out
      x_bh'  = -3 x_bh - q_bh_out
      q_bh_out = etaH * x_bh * exp[-(z/zR_bh)^2]  (if zR_bh<=0 -> window=1)
  - Flat FRW: E^2 = Ω_r0 a^-4 + Ω_b0 a^-3 + x_chi + x_phi + x_bh

Observables:
  - E(z) = H(z)/H0 (from integration)
  - D_M(z) = (c/H0) ∫ dz / E(z)
  - SN: μ_th(z) = 5 log10[(1+z) D_M(z)] + 25   (dμ handled analytically by default)
  - BAO: predict D_M(z) or D_M(z)/r_d

Outputs JSON with chi² breakdown and fit details.

Usage examples (diag SN, scaled BAO):
  python3 fit_ids_bh_diag_v2.py \
    --sn-csv data/pantheon_plus/sn_MATCHED_mu.csv \
    --sn-weight diag --sn-const-sigma 3.374 \
    --bao-csv data/desi_dr1_bao/bao_DM_over_rd.csv \
    --bao-cov data/desi_dr1_bao/bao_cov_DM_over_rd.csv \
    --bao-cov-kind cov --bao-cov-scale 200 \
    --bao-kind dm_over_rd \
    --free-Om --Om-bounds=0.25,0.65 \
    --free-rd --rd-bounds=140,180 \
    --out outputs/run.json
"""

from __future__ import annotations
import argparse, json, math, os, sys, time
from typing import Optional, Tuple, List, Dict, Callable

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, solve
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

# ----------------------------- constants -----------------------------

c_over_H0_100 = 2997.92458  # [Mpc] for H0 = 100 km/s/Mpc

def _ts(msg: str) -> str:
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"

def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5*(M + M.T)

# -------------------------- linear algebra ---------------------------

def load_spd_any(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    if path.endswith(".npy"):
        return np.load(path)
    # CSV / TXT fallback
    return pd.read_csv(path, header=None).to_numpy(float)

def cholesky_from_cov_or_prec(M: Optional[np.ndarray], kind: str, scale: float = 1.0):
    """
    Return (L, is_prec).
      - If kind='cov':  C = (S^2 * M) = L L^T, and we report is_prec=False
      - If kind='prec': P = (S^2 * M) = L^T L, and we report is_prec=True
    Here 'scale' is S (we multiply the matrix by S^2 before the factorization).
    """
    if M is None:
        return None, False
    M = np.array(M, float)
    if scale is not None and scale != 1.0:
        M = float(scale)**2 * M
    if kind == "cov":
        C = _sym(M) + 1e-12*np.eye(M.shape[0])
        L = cholesky(C)
        return L, False
    elif kind == "prec":
        P = _sym(M) + 1e-12*np.eye(M.shape[0])
        # For precision, want P = L^T L  -> choose R = chol(P), then set L = R^T so that P = L^T L
        R = cholesky(P)
        L = R.T
        return L, True
    else:
        raise ValueError("kind must be 'cov' or 'prec'")

def parse_bounds(s: Optional[str], default: Tuple[float,float]) -> Tuple[float,float]:
    if s is None: return default
    a,b = [float(t) for t in s.split(",")]
    return (a,b)

# ---------------------------- data loaders ---------------------------

def load_sn(sn_csv: str,
            sn_cov: Optional[str],
            sn_cov_kind: str,
            sn_scale: float,
            sn_weight_mode: str,
            sn_const_sigma: Optional[float]):
    """
    Returns (z, mu, L, is_prec, info_dict)
      - If sn_weight_mode == 'auto': uses provided covariance/precision
      - If sn_weight_mode == 'diag': uses constant sigma = sn_const_sigma (required)
    """
    df = pd.read_csv(sn_csv)
    z  = df["z"].to_numpy(float)
    mu = df["mu"].to_numpy(float)

    info = {"sn_weight_mode": sn_weight_mode, "sn_cov_kind_used": None, "sn_scale_used": None}

    if sn_weight_mode == "diag":
        if sn_const_sigma is None:
            raise ValueError("sn-weight=diag requires --sn-const-sigma")
        sig = np.full_like(z, float(sn_const_sigma))
        C = np.diag(sig**2)
        L, is_prec = cholesky_from_cov_or_prec(C, "cov", scale=1.0)
        info["sn_cov_kind_used"] = "cov"
        info["sn_scale_used"] = 1.0
        return z, mu, L, is_prec, info

    # auto mode
    if not sn_cov:
        # fall back to per-row sigma if present
        if "sigma" in df.columns:
            sig = df["sigma"].to_numpy(float)
            C = np.diag(sig**2)
            L, is_prec = cholesky_from_cov_or_prec(C, "cov", scale=1.0)
            info["sn_cov_kind_used"] = "cov"
            info["sn_scale_used"] = 1.0
            return z, mu, L, is_prec, info
        raise ValueError("sn-weight=auto requires --sn-cov or a 'sigma' column in the SN CSV")

    M = load_spd_any(sn_cov)
    L, is_prec = cholesky_from_cov_or_prec(M, sn_cov_kind, scale=sn_scale)
    info["sn_cov_kind_used"] = sn_cov_kind
    info["sn_scale_used"] = float(sn_scale)
    return z, mu, L, is_prec, info

def load_bao(bao_csv: Optional[str],
             bao_cov: Optional[str],
             bao_cov_kind: str,
             bao_cov_scale: float):
    """
    Returns dict or None:
      {"z": z, "y": y, "L": L, "is_prec": is_prec, "n": n}
    Interprets bao_cov_kind as 'cov' or 'prec'. Applies (bao_cov_scale)^2 to the matrix.
    If bao_cov is None, uses diagonal with sigma from CSV (column 'sigma' required).
    """
    if not bao_csv:
        return None
    df = pd.read_csv(bao_csv)
    z = df["z"].to_numpy(float)
    y = df["y"].to_numpy(float)
    if bao_cov:
        Mb = load_spd_any(bao_cov)
        Lb, is_prec = cholesky_from_cov_or_prec(Mb, bao_cov_kind, scale=bao_cov_scale)
    else:
        if "sigma" not in df.columns:
            raise ValueError("BAO CSV requires 'sigma' if --bao-cov not provided")
        sig = df["sigma"].to_numpy(float)
        Cb = np.diag(sig**2)
        Lb, is_prec = cholesky_from_cov_or_prec(Cb, "cov", scale=1.0)
    return {"z": z, "y": y, "L": Lb, "is_prec": is_prec, "n": len(z)}

# ---------------------- interacting background ----------------------

def window_gauss(z: np.ndarray, zR: float) -> np.ndarray:
    if zR is None or zR <= 0.0:
        return np.ones_like(z)
    return np.exp(-(z/zR)**2)

def integrate_background(zmax: float,
                         Om0: float,
                         w_de: float,
                         epsH: float,
                         xiH: float,
                         zR_ids: float,
                         Ob0: float,
                         Or0: float,
                         Obh0: float,
                         etaH: float,
                         fde: float,
                         zR_bh: Optional[float],
                         n_steps: int = 4000):
    """
    Integrate x_chi, x_phi, x_bh (dimensionless densities at a=1 normalized to rho_c0)
    from a=1 to a_min = 1/(1+zmax) using Heun's method (improved Euler).
    """
    a_min = 1.0/(1.0 + float(zmax))
    ln_a0, ln_amin = 0.0, math.log(a_min)
    N = max(1000, int(n_steps))
    dln = (ln_amin - ln_a0)/N  # negative

    # Initial conditions at a=1 (flatness)
    x_chi = float(Om0)
    x_bh  = float(Obh0)
    x_phi = max(1.0 - Om0 - Ob0 - Or0 - x_bh, 1e-18)

    ln_list = [ln_a0]
    z_list  = [0.0]
    E2_list = [Or0 + Ob0 + x_chi + x_phi + x_bh]

    for _ in range(N):
        ln_a = ln_list[-1]
        a = math.exp(ln_a)
        z = 1.0/a - 1.0

        W_ids = math.exp(-(z/zR_ids)**2) if zR_ids > 0 else 1.0
        W_bh  = 1.0 if (zR_bh is None or zR_bh <= 0.0) else math.exp(-(z/zR_bh)**2)

        q_ids = (epsH * x_chi + xiH * x_phi) * W_ids
        q_bh  = etaH * x_bh * W_bh

        # k1
        k1_chi = -3.0*x_chi + q_ids + (1.0 - fde)*q_bh
        k1_phi = -3.0*(1.0 + w_de)*x_phi - q_ids + fde*q_bh
        k1_bh  = -3.0*x_bh - q_bh

        # predictor
        x_chi_p = max(x_chi + dln*k1_chi, 1e-24)
        x_phi_p = max(x_phi + dln*k1_phi, 1e-24)
        x_bh_p  = max(x_bh  + dln*k1_bh,  1e-24)

        a_p = math.exp(ln_a + dln)
        z_p = 1.0/a_p - 1.0
        W_ids_p = math.exp(-(z_p/zR_ids)**2) if zR_ids > 0 else 1.0
        W_bh_p  = 1.0 if (zR_bh is None or zR_bh <= 0.0) else math.exp(-(z_p/zR_bh)**2)

        q_ids_p = (epsH * x_chi_p + xiH * x_phi_p) * W_ids_p
        q_bh_p  = etaH * x_bh_p * W_bh_p

        # k2
        k2_chi = -3.0*x_chi_p + q_ids_p + (1.0 - fde)*q_bh_p
        k2_phi = -3.0*(1.0 + w_de)*x_phi_p - q_ids_p + fde*q_bh_p
        k2_bh  = -3.0*x_bh_p - q_bh_p

        # corrector
        x_chi = max(x_chi + 0.5*dln*(k1_chi + k2_chi), 1e-24)
        x_phi = max(x_phi + 0.5*dln*(k1_phi + k2_phi), 1e-24)
        x_bh  = max(x_bh  + 0.5*dln*(k1_bh  + k2_bh),  1e-24)

        ln_new = ln_a + dln
        a_new = math.exp(ln_new)
        z_new = 1.0/a_new - 1.0
        E2 = Or0*(a_new**-4) + Ob0*(a_new**-3) + x_chi + x_phi + x_bh

        ln_list.append(ln_new)
        z_list.append(z_new)
        E2_list.append(max(E2, 1e-18))

    z_grid = np.array(z_list[::-1], float)
    E_grid = np.sqrt(np.array(E2_list[::-1], float))
    invE   = 1.0 / np.clip(E_grid, 1e-12, None)
    I_grid = cumulative_trapezoid(invE, z_grid, initial=0.0)  # dimensionless ∫dz/E
    return z_grid, E_grid, I_grid

def build_background(zmax, H0, Om0, w_de, epsH, xiH, zR_ids, Ob0, Or0,
                     Obh0, etaH, fde, zR_bh, n_steps=4000):
    z_bg, E_bg, I_bg = integrate_background(zmax, Om0, w_de, epsH, xiH, zR_ids,
                                            Ob0, Or0, Obh0, etaH, fde, zR_bh,
                                            n_steps=n_steps)
    def E_at(z):
        z = np.asarray(z, float)
        return np.interp(z, z_bg, E_bg)
    def Dc_at(z):
        z = np.asarray(z, float)
        I = np.interp(z, z_bg, I_bg)
        return (c_over_H0_100 / (H0/100.0)) * I
    return z_bg, E_at, Dc_at

# ---------------------------- predictions ---------------------------

def mu_theory(z, H0, Om0, w_de, epsH, xiH, zR_ids, Ob0, Or0,
              Obh0, etaH, fde, zR_bh, n_steps=4000):
    z = np.asarray(z, float)
    zmax = float(np.max(z)) + 0.05
    _, _, Dc_at = build_background(zmax, H0, Om0, w_de, epsH, xiH, zR_ids,
                                   Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=n_steps)
    Dl = (1.0 + z) * Dc_at(z)
    return 5.0*np.log10(np.clip(Dl, 1e-20, None)) + 25.0

def bao_prediction(z_bao, bao_kind, H0, Om0, rd, w_de, epsH, xiH, zR_ids,
                   Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=4000):
    z_bao = np.asarray(z_bao, float)
    zmax = float(np.max(z_bao)) + 0.05
    _, _, Dc_at = build_background(zmax, H0, Om0, w_de, epsH, xiH, zR_ids,
                                   Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=n_steps)
    DM = Dc_at(z_bao)
    if bao_kind == "dm_over_rd":
        return DM / rd
    elif bao_kind == "dm":
        return DM
    else:
        raise ValueError("bao_kind must be 'dm_over_rd' or 'dm'")

# ------------------------------- chi2 -------------------------------

def make_whitener(L: np.ndarray, is_prec: bool) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return f(v) such that ||f(v)||^2 = v^T C^{-1} v  (if L from covariance)
                              or     = v^T P v      (if L from precision)
    - For cov: L L^T = C; whiten(v) = solve(L, v)
    - For prec: P = L^T L; whiten(v) = L @ v
    """
    if is_prec:
        return lambda v: L @ v
    return lambda v: solve(L, v)

def make_sn_chi2(z, mu, L, is_prec, analytic_dmu: bool,
                 H0, Ob0, Or0, n_steps):
    """
    Returns chi2(theta), dmu_hat(theta) closure.
    If analytic_dmu=False, dmu_hat is always 0 here (external handling).
    """
    z = np.asarray(z, float); mu = np.asarray(mu, float)
    whiten = make_whitener(L, is_prec)
    u = np.ones_like(mu)

    def chi2(params: Dict[str,float]) -> Tuple[float, float]:
        mu_th = mu_theory(z, H0, params["Om0"], params["w_de"], params["epsH"], params["xiH"],
                          params["zR"], params["Ob0"], params["Or0"],
                          params["Obh0"], params["etaH"], params["fde"], params["zR_bh"],
                          n_steps=n_steps)
        r0 = mu - mu_th  # WITHOUT dmu; we handle dmu here if analytic
        if not analytic_dmu:
            r = r0 - params["dmu"]
            y = whiten(r)
            return float(y @ y), 0.0
        y_u = whiten(u)
        y_r = whiten(r0)
        a = float(y_u @ y_u)
        b = float(y_u @ y_r)
        if a <= 0.0:  # shouldn't happen
            chi2_val = float(y_r @ y_r)
            return chi2_val, 0.0
        dmu_hat = b / a
        y = y_r - (b/a) * y_u
        chi2_val = float(y @ y)
        return chi2_val, dmu_hat

    return chi2

def make_bao_chi2(bao_pack, bao_kind, H0, Ob0, Or0, n_steps):
    if bao_pack is None:
        def zero(params): return 0.0
        return zero
    z = bao_pack["z"]; y = bao_pack["y"]; L = bao_pack["L"]; is_prec = bao_pack["is_prec"]
    whiten = make_whitener(L, is_prec)
    def chi2(params: Dict[str, float]) -> float:
        y_th = bao_prediction(z, bao_kind, H0, params["Om0"], params["rd"],
                              params["w_de"], params["epsH"], params["xiH"], params["zR"],
                              params["Ob0"], params["Or0"], params["Obh0"], params["etaH"],
                              params["fde"], params["zR_bh"], n_steps=n_steps)
        r = y - y_th
        yr = whiten(r)
        return float(yr @ yr)
    return chi2

# ------------------------------- main --------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--sn-csv", required=True)
    ap.add_argument("--sn-cov", default=None)
    ap.add_argument("--sn-cov-kind", choices=["cov","prec"], default="cov")
    ap.add_argument("--sn-scale", type=float, default=1.0)
    ap.add_argument("--sn-const-sigma", type=float, default=None)
    ap.add_argument("--sn-weight", choices=["auto","diag"], default="auto")

    ap.add_argument("--bao-csv", default=None)
    ap.add_argument("--bao-cov", default=None)
    ap.add_argument("--bao-cov-kind", choices=["cov","prec"], default="cov")
    ap.add_argument("--bao-cov-scale", type=float, default=1.0)
    ap.add_argument("--bao-kind", choices=["dm_over_rd","dm"], default="dm_over_rd")

    # Background reference
    ap.add_argument("--fid-H0", dest="H0", type=float, default=70.0)
    ap.add_argument("--fid-Om", dest="Om0", type=float, default=0.3)

    # Present-day fixed (flat)
    ap.add_argument("--Omega_b0", type=float, default=0.0)
    ap.add_argument("--Omega_r0", type=float, default=0.0)

    # IDS params
    ap.add_argument("--w_de", type=float, default=-1.0)
    ap.add_argument("--epsH", type=float, default=0.0)
    ap.add_argument("--xiH",  type=float, default=0.0)
    ap.add_argument("--zR",   type=float, default=0.6)

    # BAO rd and SN dmu
    ap.add_argument("--rd",   type=float, default=147.1)
    ap.add_argument("--analytic-dmu", dest="analytic_dmu", action="store_true", default=True)
    ap.add_argument("--no-analytic-dmu", dest="analytic_dmu", action="store_false")
    ap.add_argument("--dmu",  type=float, default=0.0)

    # BH reservoir params
    ap.add_argument("--Omega_bh0", type=float, default=0.0)
    ap.add_argument("--etaH", type=float, default=0.0)
    ap.add_argument("--fde",  type=float, default=1.0)
    ap.add_argument("--zR_bh", type=float, default=None)

    # Bounds
    ap.add_argument("--Om-bounds", default="0.05,0.70")
    ap.add_argument("--wde-bounds", default="-1.2,-0.8")
    ap.add_argument("--epsH-bounds", default="-0.05,0.05")
    ap.add_argument("--xiH-bounds",  default="-0.05,0.05")
    ap.add_argument("--zR-bounds",   default="0.2,1.2")
    ap.add_argument("--rd-bounds",   default=None)
    ap.add_argument("--dmu-bounds",  default=None)
    ap.add_argument("--obh-bounds",  default="0.0,0.01")
    ap.add_argument("--etaH-bounds", default="0.0,0.2")
    ap.add_argument("--fde-bounds",  default="0.0,1.0")
    ap.add_argument("--zRbh-bounds", default="0.2,1.5")

    # Free flags
    ap.add_argument("--free-Om", action="store_true")
    ap.add_argument("--free-wde", action="store_true")
    ap.add_argument("--free-epsH", action="store_true")
    ap.add_argument("--free-xiH", action="store_true")
    ap.add_argument("--free-zR", action="store_true")
    ap.add_argument("--free-rd", action="store_true")
    ap.add_argument("--free-dmu", action="store_true")
    ap.add_argument("--free-obh", action="store_true")
    ap.add_argument("--free-etaH", action="store_true")
    ap.add_argument("--free-fde", action="store_true")
    ap.add_argument("--free-zRbh", action="store_true")

    # Controls
    ap.add_argument("--bg-steps", type=int, default=4000)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--verbose-every", type=int, default=25)
    ap.add_argument("--out", default="outputs/ids_fit.json")

    args = ap.parse_args()

    # Load SN and BAO
    z_sn, mu_sn, L_sn, sn_is_prec, sn_info = load_sn(
        args.sn_csv, args.sn_cov, args.sn_cov_kind, args.sn_scale,
        args.sn_weight, args.sn_const_sigma
    )

    bao_pack = load_bao(args.bao_csv, args.bao_cov, args.bao_cov_kind, args.bao_cov_scale) if args.bao_csv else None

    # Build chi2 closures
    sn_chi2_fn = make_sn_chi2(z_sn, mu_sn, L_sn, sn_is_prec, args.analytic_dmu,
                              args.H0, args.Omega_b0, args.Omega_r0, n_steps=args.bg_steps)
    bao_chi2_fn = make_bao_chi2(bao_pack, args.bao_kind, args.H0, args.Omega_b0, args.Omega_r0, n_steps=args.bg_steps)

    # Defaults
    defaults = dict(
        H0=float(args.H0),
        Om0=float(args.Om0),
        w_de=float(args.w_de),
        epsH=float(args.epsH),
        xiH=float(args.xiH),
        zR=float(args.zR),
        rd=float(args.rd),
        dmu=float(args.dmu),
        Ob0=float(args.Omega_b0),
        Or0=float(args.Omega_r0),
        Obh0=float(args.Omega_bh0),
        etaH=float(args.etaH),
        fde=float(args.fde),
        zR_bh=(None if args.zR_bh in [None, "", "None"] else float(args.zR_bh)),
    )

    names: List[str] = []
    bounds: List[Tuple[float,float]] = []
    x0: List[float] = []

    def add(name, val, bnd, free: bool):
        if free:
            names.append(name); bounds.append(bnd); x0.append(val)

    add("Om0", defaults["Om0"], parse_bounds(args.Om_bounds, (0.05,0.70)), bool(args.free_Om))
    add("w_de", defaults["w_de"], parse_bounds(args.wde_bounds, (-1.2,-0.8)), bool(args.free_wde))
    add("epsH", defaults["epsH"], parse_bounds(args.epsH_bounds, (-0.05,0.05)), bool(args.free_epsH))
    add("xiH",  defaults["xiH"],  parse_bounds(args.xiH_bounds,  (-0.05,0.05)), bool(args.free_xiH))
    add("zR",   defaults["zR"],   parse_bounds(args.zR_bounds,   (0.2,1.2)),    bool(args.free_zR))
    if args.free_rd:
        rd_b = parse_bounds(args.rd_bounds, (120.0, 170.0)) if args.rd_bounds else (120.0,170.0)
        add("rd", defaults["rd"], rd_b, True)
    if args.free_dmu and not args.analytic_dmu:
        dmu_b = parse_bounds(args.dmu_bounds, (-2.0,2.0)) if args.dmu_bounds else (-2.0,2.0)
        add("dmu", defaults["dmu"], dmu_b, True)

    add("Obh0", defaults["Obh0"], parse_bounds(args.obh_bounds, (0.0,0.01)), bool(args.free_obh))
    add("etaH", defaults["etaH"], parse_bounds(args.etaH_bounds, (0.0,0.2)), bool(args.free_etaH))
    add("fde",  defaults["fde"],  parse_bounds(args.fde_bounds,  (0.0,1.0)),  bool(args.free_fde))
    if args.free_zRbh:
        zrb = parse_bounds(args.zRbh_bounds, (0.2,1.5))
        add("zR_bh", float(0.8 if defaults["zR_bh"] is None else defaults["zR_bh"]), zrb, True)

    def theta_to_params(theta):
        p = dict(defaults)
        for k,v in zip(names, theta): p[k] = float(v)
        return p

    it_counter = {"n": 0}
    def total_chi2(theta):
        p = theta_to_params(theta)
        c2_sn, dmu_hat = sn_chi2_fn(p)
        p["_dmu_hat"] = dmu_hat
        c2_bao = bao_chi2_fn(p)
        return c2_sn + c2_bao

    def cb(xk):
        it_counter["n"] += 1
        n = it_counter["n"]
        if args.verbose_every and n % args.verbose_every == 0:
            print(_ts(f"eval {n}: chi2={total_chi2(xk):.3f}"), file=sys.stderr, flush=True)

    t0 = time.time()
    res = minimize(total_chi2, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=args.maxiter), callback=cb)
    elapsed = time.time() - t0

    best = theta_to_params(x0)
    if res.success:
        best = theta_to_params(list(res.x))

    # Evaluate breakdown with final dmu_hat
    c2_sn, dmu_hat = sn_chi2_fn(best)
    c2_bao = bao_chi2_fn(best)
    c2_tot = c2_sn + c2_bao

    # Bounds hits
    hit = {}
    for nm,(lo,hi) in zip(names, bounds):
        val = best[nm]
        hit[nm] = bool(abs(val-lo) < 1e-12 or abs(val-hi) < 1e-12)

    ndof = (len(z_sn) + (bao_pack["n"] if bao_pack else 0)) - len(names)

    out = {
        "best_params": {
            "H0": float(best["H0"]),
            "Om0": float(best["Om0"]),
            "w_de": float(best["w_de"]),
            "epsH": float(best["epsH"]),
            "xiH":  float(best["xiH"]),
            "zR":   float(best["zR"]),
            "rd":   float(best["rd"]),
            "dmu":  (float(best["dmu"]) if (not args.analytic_dmu and "dmu" in names) else float(args.dmu)),
            "Ob0":  float(best["Ob0"]),
            "Or0":  float(best["Or0"]),
            "Obh0": float(best["Obh0"]),
            "etaH": float(best["etaH"]),
            "fde":  float(best["fde"]),
            "zR_bh": (None if best["zR_bh"] is None else float(best["zR_bh"])),
            "analytic_dmu": bool(args.analytic_dmu)
        },
        "fit_params": names,
        "chi2": float(c2_tot),
        "ndof": int(ndof),
        "success": bool(res.success),
        "message": (str(res.message) if hasattr(res, "message") else ""),
        "breakdown": {"chi2_bao": float(c2_bao), "chi2_sn": float(c2_sn)},
        "timing_sec": float(elapsed),
        "sn_weight_mode": args.sn_weight,
        "sn_cov_kind_used": sn_info["sn_cov_kind_used"],
        "sn_scale_used": sn_info["sn_scale_used"],
        "bao_kind_used": args.bao_kind,
        "bao_cov_kind_used": args.bao_cov_kind,
        "bao_cov_scale_used": float(args.bao_cov_scale),
        "sn_points": int(len(z_sn)),
        "bao_points": int(bao_pack["n"] if bao_pack else 0),
        "hit_bounds": hit,
        "evals": int(it_counter["n"]),
        "config": {
            "bounds": {
                "Om0": args.Om_bounds, "w_de": args.wde_bounds, "epsH": args.epsH_bounds, "xiH": args.xiH_bounds,
                "zR": args.zR_bounds, "rd": args.rd_bounds, "dmu": args.dmu_bounds,
                "obh": args.obh_bounds, "etaH": args.etaH_bounds, "fde": args.fde_bounds, "zR_bh": args.zRbh_bounds
            },
            "free": {
                "Om0": bool(args.free_Om), "w_de": bool(args.free_wde),
                "epsH": bool(args.free_epsH), "xiH": bool(args.free_xiH),
                "zR": bool(args.free_zR), "rd": bool(args.free_rd), "dmu": bool(args.free_dmu),
                "Obh0": bool(args.free_obh), "etaH": bool(args.free_etaH), "fde": bool(args.free_fde),
                "zR_bh": bool(args.free_zRbh)
            }
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
