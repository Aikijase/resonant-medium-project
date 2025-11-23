#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interacting "Resonant Rhythm" Dark Sector fitter for SN + BAO, with optional
Black Hole (BH) Hawking feedback reservoir.

Key ideas:
  - Effective fluids in flat FRW: x_chi (DM-like), x_phi (DE-like), optional x_bh (BH mass).
  - Base interaction (windowed by redshift):
        q_base = (epsH * x_chi + xiH * x_phi) * exp[-(z/zR)^2]
        x_chi' = -3 x_chi + q_base
        x_phi' = -3 (1+w_de) x_phi - q_base
  - BH Hawking "permeation" loop (optional; off by default unless Omega_bh0>0 and/or etaH>0):
        q_bh   = etaH * x_bh * exp[-(z/zR_bh)^2]
        x_bh'  = -3 x_bh - q_bh
        x_chi' += (1-fde) * q_bh
        x_phi' += fde * q_bh

Observables:
  - E(z) = H(z)/H0
  - D_M(z) = (c/H0) ∫ dz / E(z)
  - SN: mu(z) = 5 log10[(1+z) D_M(z)] + 25 (+ dmu* analytically solved each eval if --analytic-dmu)
  - BAO: DM(z) or DM(z)/r_d

Notes:
  - Analytic dmu is used by default (stable zero-point); you can disable with --no-analytic-dmu and fit dmu.
  - Supports SN covariance (cov) or precision (prec). If omitted and the SN CSV has a "sigma" column,
    a diagonal covariance is constructed. You can also force a constant-diagonal via --sn-const-sigma.
  - BAO covariance optional; if absent, expects a "sigma" column in the BAO CSV to build a diagonal.
  - Includes flatness sanity penalty if Omega sums exceed 1 at a=1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import lru_cache
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from numpy.linalg import cholesky
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

# ----------------------------- utils -----------------------------

c_km_s = 299792.458
c_over_H0_100 = 2997.92458  # Mpc for H0 = 100 km/s/Mpc

def _ts(msg: str) -> str:
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"

def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5*(M + M.T)

def load_spd_any(path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load a symmetric positive (semi)definite matrix from .npy (dense) or CSV.
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        return np.load(path)
    # tolerate commas or whitespace
    try:
        return pd.read_csv(path, header=None).to_numpy(dtype=float)
    except Exception:
        return np.loadtxt(path, dtype=float)

def cholesky_from_cov_or_prec(M: Optional[np.ndarray], kind: str, scale: float = 1.0):
    """Return (L, is_prec).
       If kind == 'cov': C = L L^T (L lower-tri Cholesky of covariance)
       If kind == 'prec': P = L^T L (L upper-tri such that P = L^T L)"""
    if M is None:
        return None, False
    M = np.array(M, float)
    if scale is not None and scale != 1.0:
        M = float(scale) * M
    if kind == "cov":
        C = _sym(M) + 1e-12*np.eye(M.shape[0])
        L = cholesky(C)
        return L, False
    elif kind == "prec":
        P = _sym(M) + 1e-12*np.eye(M.shape[0])
        # For precision, we want P = L^T L; take the standard lower-tri of P, then transpose.
        L_lower = cholesky(P)
        L = L_lower.T
        return L, True
    else:
        raise ValueError("kind must be 'cov' or 'prec'")

def parse_bounds(s: Optional[str], default: Tuple[float,float]) -> Tuple[float,float]:
    if s is None: return default
    a,b = [float(t) for t in s.split(",")]
    return (a,b)

def detect_sigma_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["sigma","sig","mu_sigma","sigma_mu","sigma_tot","muerr","dmu_err","mu_err","yerr","err","unc"]:
        if c in df.columns:
            return c
    return None

# ------------------------- data loaders --------------------------

def load_sn(sn_csv: str,
            sn_cov: Optional[str],
            cov_kind: str,
            scale: float,
            sn_const_sigma: Optional[float] = None):
    df = pd.read_csv(sn_csv)
    if "z" not in df.columns or "mu" not in df.columns:
        raise ValueError("SN CSV must contain columns 'z' and 'mu'")
    z  = df["z"].to_numpy(float)
    mu = df["mu"].to_numpy(float)

    # Covariance logic
    L = None; is_prec = False
    if sn_const_sigma is not None:
        # construct constant diagonal covariance
        C = np.eye(len(df), dtype=float) * (float(sn_const_sigma)**2)
        L, is_prec = cholesky_from_cov_or_prec(C, "cov", scale=1.0)
    elif sn_cov:
        M = load_spd_any(sn_cov)
        L, is_prec = cholesky_from_cov_or_prec(M, cov_kind, scale=scale)
    else:
        col = detect_sigma_column(df)
        if col is None:
            raise ValueError("No SN covariance provided and no sigma column found to construct diagonal.")
        sig = df[col].to_numpy(float)
        C = np.diag(sig**2)
        L, is_prec = cholesky_from_cov_or_prec(C, "cov", scale=1.0)
    return z, mu, L, is_prec

def load_bao(bao_csv: Optional[str], bao_cov: Optional[str]):
    if not bao_csv:
        return None
    df = pd.read_csv(bao_csv)
    if "z" not in df.columns or "y" not in df.columns:
        raise ValueError("BAO CSV must contain columns 'z' and 'y' (where y is DM or DM/rd)")
    z = df["z"].to_numpy(float)
    y = df["y"].to_numpy(float)

    # covariance or diagonal
    if bao_cov:
        Mb = load_spd_any(bao_cov)
        Lb, is_prec = cholesky_from_cov_or_prec(Mb, "cov", scale=1.0)  # BAO cov assumed 'cov'
    else:
        col = detect_sigma_column(df)
        if col is None:
            raise ValueError("No BAO covariance provided and no sigma column found in BAO CSV.")
        sig = df[col].to_numpy(float)
        Cb = np.diag(sig**2)
        Lb, is_prec = cholesky_from_cov_or_prec(Cb, "cov", scale=1.0)
    return {"z": z, "y": y, "L": Lb, "is_prec": False, "n": len(z)}

# ---------------------- interacting background -------------------

def window_gauss_z(z: float, zR: float) -> float:
    if zR is None or zR <= 0:
        return 1.0
    return math.exp(- (z/zR)**2 )

@lru_cache(maxsize=256)
def _bg_raw(zmax: float,
            H0: float,
            Om0: float,
            w_de: float,
            epsH: float,
            xiH: float,
            zR: float,
            Ob0: float,
            Or0: float,
            Obh0: float,
            etaH: float,
            fde: float,
            zR_bh: Optional[float],
            n_steps: int):
    """
    Integrate x_chi(a), x_phi(a), x_bh(a) from a=1 to a_min given parameters.
    Returns z_grid (ascending), E_grid, Dc_integral_grid (dimensionless ∫dz/E).
    """
    a_min = 1.0/(1.0 + float(zmax))
    ln_a0, ln_amin = 0.0, math.log(a_min)
    N = max(1000, int(n_steps))
    dln = (ln_amin - ln_a0)/N  # negative

    # initial conditions at a=1
    x_chi = float(Om0)  # DM-like
    x_bh  = max(float(Obh0), 0.0)
    x_phi = max(1.0 - Om0 - Ob0 - Or0 - x_bh, 1e-16)  # flatness

    ln_a_list = [ln_a0]
    a_list = [1.0]
    z_list = [0.0]
    E2_list = [Or0 + Ob0 + x_chi + x_phi + x_bh]

    for _ in range(N):
        ln_a = ln_a_list[-1]
        a = math.exp(ln_a)
        z = 1.0/a - 1.0

        W  = window_gauss_z(z, zR)
        Wb = window_gauss_z(z, (zR if (zR_bh is None) else zR_bh))

        # sources
        q_base1 = (epsH * x_chi + xiH * x_phi) * W
        q_bh1   = (etaH * x_bh) * Wb

        k1_chi = -3.0*x_chi + q_base1 + (1.0 - fde) * q_bh1
        k1_phi = -3.0*(1.0 + w_de)*x_phi - q_base1 + fde * q_bh1
        k1_bh  = -3.0*x_bh  - q_bh1

        # predictor
        x_chi_p = max(x_chi + dln*k1_chi, 1e-18)
        x_phi_p = max(x_phi + dln*k1_phi, 1e-18)
        x_bh_p  = max(x_bh  + dln*k1_bh,  1e-18)

        a_p = math.exp(ln_a + dln)
        z_p = 1.0/a_p - 1.0
        Wp  = window_gauss_z(z_p, zR)
        Wbp = window_gauss_z(z_p, (zR if (zR_bh is None) else zR_bh))

        q_base2 = (epsH * x_chi_p + xiH * x_phi_p) * Wp
        q_bh2   = (etaH * x_bh_p) * Wbp

        k2_chi = -3.0*x_chi_p + q_base2 + (1.0 - fde) * q_bh2
        k2_phi = -3.0*(1.0 + w_de)*x_phi_p - q_base2 + fde * q_bh2
        k2_bh  = -3.0*x_bh_p  - q_bh2

        # corrector
        x_chi = max(x_chi + 0.5*dln*(k1_chi + k2_chi), 1e-18)
        x_phi = max(x_phi + 0.5*dln*(k1_phi + k2_phi), 1e-18)
        x_bh  = max(x_bh  + 0.5*dln*(k1_bh  + k2_bh ), 1e-18)

        ln_a_new = ln_a + dln
        a_new = math.exp(ln_a_new)
        z_new = 1.0/a_new - 1.0

        E2_new = Or0*(a_new**-4) + Ob0*(a_new**-3) + x_chi + x_phi + x_bh
        E2_list.append(max(E2_new, 1e-16))
        ln_a_list.append(ln_a_new)
        a_list.append(a_new)
        z_list.append(z_new)

    # ascending z
    z_grid = np.array(z_list[::-1])
    E_grid = np.sqrt(np.array(E2_list[::-1], float))
    invE = 1.0 / np.clip(E_grid, 1e-12, None)
    Dc_over_cH0 = cumulative_trapezoid(invE, z_grid, initial=0.0)  # dimensionless ∫dz/E
    return z_grid, E_grid, Dc_over_cH0

def build_background_and_interps(zmax, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=4000):
    z_bg, E_bg, I_bg = _bg_raw(float(zmax), float(H0), float(Om0), float(w_de),
                               float(epsH), float(xiH), float(zR),
                               float(Ob0), float(Or0), float(Obh0),
                               float(etaH), float(fde),
                               (None if zR_bh is None else float(zR_bh)),
                               int(n_steps))
    def E_at(z):
        z = np.asarray(z, float)
        return np.interp(z, z_bg, E_bg)
    def Dc_at(z):
        z = np.asarray(z, float)
        I = np.interp(z, z_bg, I_bg)
        return (c_over_H0_100 / (H0/100.0)) * I
    return z_bg, E_at, Dc_at

def mu_theory(z, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=4000):
    z = np.asarray(z, float)
    zmax = float(np.max(z)) + 0.05
    _, _, Dc_at = build_background_and_interps(zmax, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=n_steps)
    Dl = (1.0 + z) * Dc_at(z)
    return 5.0*np.log10(np.clip(Dl, 1e-20, None)) + 25.0

def bao_predict(z_bao, bao_kind, H0, Om0, rd, w_de, epsH, xiH, zR, Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=4000):
    z_bao = np.asarray(z_bao, float)
    zmax = float(np.max(z_bao)) + 0.05
    _, _, Dc_at = build_background_and_interps(zmax, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, Obh0, etaH, fde, zR_bh, n_steps=n_steps)
    DM = Dc_at(z_bao)
    if bao_kind == "dm_over_rd":
        return DM / rd
    elif bao_kind == "dm":
        return DM
    else:
        raise ValueError("bao_kind must be 'dm_over_rd' or 'dm'")

# ------------------------------ chi2 ------------------------------

def make_sn_chi2(z, mu, L, is_prec, H0, Ob0, Or0, analytic_dmu: bool, n_steps):
    z = np.asarray(z, float); mu = np.asarray(mu, float)
    ones = np.ones_like(mu)

    def quad_weighted(res):
        # if precision P = L^T L → ||L res||^2 ; if covariance C = L L^T → ||L^{-1} res||^2
        if is_prec:
            y = L @ res
        else:
            y = solve_triangular(L, res, lower=True, check_finite=False)
        return float(y @ y)

    # Precompute C^{-1} 1 or helpers for analytic dmu
    if analytic_dmu:
        if is_prec:
            v = L @ ones
            P1 = L.T @ v
            denom = float(ones @ P1)
            def dmu_star(res0):
                # num = 1^T P res0 = (L 1)·(L res0)
                num = float((L @ ones) @ (L @ res0))
                return num / denom
        else:
            # Solve C x = 1 → L L^T x = 1
            u = solve_triangular(L, ones, lower=True, check_finite=False)
            x = solve_triangular(L.T, u, lower=False, check_finite=False)
            P1 = x
            denom = float(ones @ P1)
            def dmu_star(res0):
                num = float(ones @ (P1 * res0))  # 1^T C^{-1} res0
                return num / denom
    else:
        dmu_star = None  # not used if analytic_dmu False

    def chi2(params: Dict[str, float]) -> float:
        mu_th = mu_theory(z, params["H0"], params["Om0"], params["w_de"],
                          params["epsH"], params["xiH"], params["zR"],
                          params["Ob0"], params["Or0"],
                          params["Obh0"], params["etaH"], params["fde"], params["zR_bh"],
                          n_steps=n_steps)
        res0 = mu - mu_th
        if analytic_dmu:
            dmu = dmu_star(res0)
            resid = res0 - dmu
            return quad_weighted(resid)
        else:
            resid = mu - (mu_th + params["dmu"])
            return quad_weighted(resid)

    return chi2

def make_bao_chi2(bao_pack, bao_kind, H0, Ob0, Or0, n_steps):
    if bao_pack is None:
        return lambda params: 0.0
    z = bao_pack["z"]; y = bao_pack["y"]; L = bao_pack["L"]; is_prec = bao_pack["is_prec"]

    def quad(res):
        # BAO cov assumed as covariance: C = L L^T
        ytri = solve_triangular(L, res, lower=True, check_finite=False)
        return float(ytri @ ytri)

    def chi2(params: Dict[str, float]) -> float:
        y_th = bao_predict(z, bao_kind, params["H0"], params["Om0"], params["rd"],
                           params["w_de"], params["epsH"], params["xiH"], params["zR"],
                           params["Ob0"], params["Or0"], params["Obh0"], params["etaH"], params["fde"], params["zR_bh"],
                           n_steps=n_steps)
        return quad(y - y_th)

    return chi2

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Interacting Rhythm Dark Sector fitter (SN+BAO) with optional BH Hawking loop")
    # Data
    ap.add_argument("--sn-csv", required=True)
    ap.add_argument("--sn-cov", default=None)
    ap.add_argument("--sn-cov-kind", choices=["cov","prec"], default="cov")
    ap.add_argument("--sn-scale", type=float, default=1.0)
    ap.add_argument("--sn-const-sigma", type=float, default=None, help="If set, build constant-diagonal SN covariance with this sigma (mag)")

    ap.add_argument("--bao-csv", default=None)
    ap.add_argument("--bao-cov", default=None)
    ap.add_argument("--bao-kind", choices=["dm_over_rd","dm"], default="dm_over_rd")

    # Background reference
    ap.add_argument("--fid-H0", dest="H0", type=float, default=70.0)

    # Present-day fractions (flat)
    ap.add_argument("--fid-Om", dest="Om0", type=float, default=0.3)
    ap.add_argument("--Omega_b0", type=float, default=0.0)
    ap.add_argument("--Omega_r0", type=float, default=0.0)

    # IDS base params
    ap.add_argument("--w_de", type=float, default=-1.0)
    ap.add_argument("--epsH", type=float, default=0.0)
    ap.add_argument("--xiH",  type=float, default=0.0)
    ap.add_argument("--zR",   type=float, default=0.6)
    ap.add_argument("--rd",   type=float, default=147.1)

    # BH loop params
    ap.add_argument("--Omega_bh0", type=float, default=0.0, help="Present BH mass density as fraction of critical today")
    ap.add_argument("--etaH",      type=float, default=0.0, help="Hawking coupling strength (dimensionless)")
    ap.add_argument("--fde",       type=float, default=1.0, help="Fraction of BH power routed to DE (rest to DM)")
    ap.add_argument("--zR_bh",     type=float, default=None, help="Redshift window scale for BH coupling; defaults to zR if None")

    # Zero-point handling
    ap.add_argument("--analytic-dmu", dest="analytic_dmu", action="store_true", help="Solve dmu analytically each evaluation")
    ap.add_argument("--no-analytic-dmu", dest="analytic_dmu", action="store_false")
    ap.set_defaults(analytic_dmu=True)
    ap.add_argument("--dmu",  type=float, default=0.0, help="Only used if --no-analytic-dmu")

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
    ap.add_argument("--free-dmu", action="store_true")  # only matters if --no-analytic-dmu
    ap.add_argument("--free-obh", action="store_true")
    ap.add_argument("--free-etaH", action="store_true")
    ap.add_argument("--free-fde", action="store_true")
    ap.add_argument("--free-zRbh", action="store_true")

    # Controls
    ap.add_argument("--bg-steps", type=int, default=4000, help="ln a steps for background integrator")
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--verbose-every", type=int, default=25)
    ap.add_argument("--out", default="outputs/ids_fit_bh.json")

    ap.add_argument("--sn-scale-note", default="", help="optional note; echoed in JSON")

    args = ap.parse_args()

    # Load data
    z_sn, mu_sn, L_sn, sn_is_prec = load_sn(args.sn_csv, args.sn_cov, cov_kind=args.sn_cov_kind, scale=args.sn_scale, sn_const_sigma=args.sn_const_sigma)
    bao_pack = load_bao(args.bao_csv, args.bao_cov) if args.bao_csv else None

    # Defaults and free vector
    defaults = dict(
        H0=float(args.H0),
        Om0=float(args.Om0),
        w_de=float(args.w_de),
        epsH=float(args.epsH),
        xiH=float(args.xiH),
        zR=float(args.zR),
        rd=float(args.rd),
        # dmu is only used if analytic_dmu is False; keep for completeness
        dmu=float(args.dmu),
        Ob0=float(args.Omega_b0),
        Or0=float(args.Omega_r0),
        Obh0=float(args.Omega_bh0),
        etaH=float(args.etaH),
        fde=float(args.fde),
        zR_bh=(None if args.zR_bh is None else float(args.zR_bh)),
        analytic_dmu=bool(args.analytic_dmu),
    )

    names: List[str] = []
    bounds: List[Tuple[float,float]] = []
    x0: List[float] = []

    def add(name, val, bnd, free: bool):
        if free:
            names.append(name)
            bounds.append(bnd)
            x0.append(val)

    add("Om0", defaults["Om0"], parse_bounds(args.Om_bounds, (0.05,0.70)), bool(args.free_Om))
    add("w_de", defaults["w_de"], parse_bounds(args.wde_bounds, (-1.2,-0.8)), bool(args.free_wde))
    add("epsH", defaults["epsH"], parse_bounds(args.epsH_bounds, (-0.05,0.05)), bool(args.free_epsH))
    add("xiH",  defaults["xiH"],  parse_bounds(args.xiH_bounds,  (-0.05,0.05)), bool(args.free_xiH))
    add("zR",   defaults["zR"],   parse_bounds(args.zR_bounds,   (0.2,1.2)),    bool(args.free_zR))
    if args.free_rd:
        rd_b = parse_bounds(args.rd_bounds, (120.0, 170.0)) if args.rd_bounds else (120.0,170.0)
        add("rd", defaults["rd"], rd_b, True)
    if (not defaults["analytic_dmu"]) and args.free_dmu:
        dmu_b = parse_bounds(args.dmu_bounds, (-2.0,2.0)) if args.dmu_bounds else (-5.0,5.0)
        add("dmu", defaults["dmu"], dmu_b, True)
    add("Obh0", defaults["Obh0"], parse_bounds(args.obh_bounds, (0.0,0.01)), bool(args.free_obh))
    add("etaH", defaults["etaH"], parse_bounds(args.etaH_bounds, (0.0,0.2)), bool(args.free_etaH))
    add("fde",  defaults["fde"],  parse_bounds(args.fde_bounds,  (0.0,1.0)),  bool(args.free_fde))
    if args.free_zRbh:
        zrb_b = parse_bounds(args.zRbh_bounds, (0.2,1.5))
        add("zR_bh", float(defaults["zR_bh"] if defaults["zR_bh"] is not None else defaults["zR"]), zrb_b, True)

    # helper to merge theta into full params
    def theta_to_params(theta):
        params = dict(defaults)
        for k,v in zip(names, theta):
            params[k] = float(v)
        return params

    sn_chi2  = make_sn_chi2(z_sn, mu_sn, L_sn, sn_is_prec, defaults["H0"], defaults["Ob0"], defaults["Or0"], defaults["analytic_dmu"], n_steps=args.bg_steps)
    bao_chi2 = make_bao_chi2(bao_pack, args.bao_kind, defaults["H0"], defaults["Ob0"], defaults["Or0"], n_steps=args.bg_steps)

    it_counter = {"n": 0}
    def total_chi2(theta):
        p = theta_to_params(theta)
        # flatness sanity: ensure DE today non-negative
        over = p["Om0"] + p["Ob0"] + p["Or0"] + p["Obh0"] - 1.0
        if over > 1e-9:
            return 1e12 + 1e10*over
        return sn_chi2(p) + bao_chi2(p)

    def cb(xk):
        it_counter["n"] += 1
        n = it_counter["n"]
        if args.verbose_every and n % args.verbose_every == 0:
            print(_ts(f"eval {n}: chi2={total_chi2(xk):.3f}"), file=sys.stderr, flush=True)

    t0 = time.time()
    if names:
        res = minimize(total_chi2, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=args.maxiter), callback=cb)
        success = bool(res.success)
        best = theta_to_params(list(res.x) if success else x0)
        message = str(res.message) if hasattr(res, "message") else ""
    else:
        # no free params: just evaluate once
        success = True
        best = theta_to_params(x0)
        message = "No free parameters; evaluated once."

    elapsed = time.time() - t0

    # Evaluate breakdown
    c2_sn = sn_chi2(best)
    c2_bao = bao_chi2(best)
    c2_tot = c2_sn + c2_bao

    # Bounds hits
    hit = {}
    for nm,(lo,hi) in zip(names, bounds):
        val = best[nm]
        hit[nm] = bool(abs(val-lo) < 1e-9 or abs(val-hi) < 1e-9)

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
            "dmu":  float(best["dmu"]),
            "Ob0":  float(best["Ob0"]),
            "Or0":  float(best["Or0"]),
            "Obh0": float(best["Obh0"]),
            "etaH": float(best["etaH"]),
            "fde":  float(best["fde"]),
            "zR_bh": (None if best["zR_bh"] is None else float(best["zR_bh"])),
            "analytic_dmu": bool(best["analytic_dmu"]),
        },
        "fit_params": names,
        "chi2": float(c2_tot),
        "ndof": int(ndof),
        "success": bool(success),
        "message": message,
        "breakdown": {"chi2_bao": float(c2_bao), "chi2_sn": float(c2_sn)},
        "timing_sec": float(elapsed),
        "sn_cov_kind_used": ("prec" if sn_is_prec else "cov"),
        "sn_scale_used": float(args.sn_scale),
        "bao_kind_used": args.bao_kind,
        "sn_scale_note": args.sn_scale_note,
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
                "Obh0": bool(args.free_obh), "etaH": bool(args.free_etaH), "fde": bool(args.free_fde), "zR_bh": bool(args.free_zRbh)
            }
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
