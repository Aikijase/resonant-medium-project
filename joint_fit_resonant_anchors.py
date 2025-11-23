#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint SN+BAO fitter with an optional 'anchor' (phase pin) tweak.

Key fixes vs older versions:
- Uses SciPy's integrators: trapezoid, cumulative_trapezoid (no np.cumtrapz/np.trapz).
- Applies dmu to SNe residuals ALWAYS (regardless of whether epsilon is free).
- Parameter unpacking is robust and order-stable.
- JSON output fully serializable (casts numpy types to Python floats/bools).
- Includes a 'config' blob in the JSON to make apples-to-apples comparisons easy.

Supported data:
- SN: Pantheon+ (mu vs z) with either a full covariance (COV) or a precision SPD (PREC).
- BAO: compressed DM/rd or DM, with covariance if provided (otherwise diagonal from 'sigma').

Free parameters (subset by flags):
  [A, f, phi, gamma, Om, rd, anchor_eps, dmu]
Anchor tweak (phase pin):
  phi_eff(z) = phi + anchor_eps * exp[-(z/anchor_zR)^2]

Author: rebuilt helper script
"""

from __future__ import annotations
import argparse, json, math, os, sys, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, solve
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import minimize

# ----------------------- utilities -----------------------

def _progress(msg: str) -> None:
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} {msg}", file=sys.stderr, flush=True)

def _symmetrize_spd(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def load_spd_any(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    if path.endswith(".npy"):
        return np.load(path)
    # csv/txt
    return pd.read_csv(path, header=None).to_numpy(dtype=float)

def cholesky_from_cov_or_prec(M: Optional[np.ndarray], kind: str, scale: float = 1.0):
    """
    Returns (L, is_prec):
      if kind == 'cov'  : C = L L^T  and quad = ||solve(L, r)||^2
      if kind == 'prec' : P = L^T L  and quad = ||L r||^2
    """
    if M is None:
        return None, False
    M = np.array(M, dtype=float)
    if scale is not None and scale != 1.0:
        M = float(scale) * M
    if kind == "cov":
        C = _symmetrize_spd(M) + 1e-12 * np.eye(M.shape[0])
        L = cholesky(C)
        return L, False
    elif kind == "prec":
        P = _symmetrize_spd(M) + 1e-12 * np.eye(M.shape[0])
        # P = L^T L  (so quad = ||L r||^2)
        L = cholesky(P).T
        return L, True
    else:
        raise ValueError("kind must be 'cov' or 'prec'")

def parse_bounds(s: Optional[str], default: Tuple[float, float]) -> Tuple[float, float]:
    if s is None:
        return default
    a, b = [float(t) for t in s.split(",")]
    return (a, b)

# -------------------- data loaders -----------------------

def load_sn(sn_csv: str, sn_cov_path: str, cov_kind: str, scale: float):
    df = pd.read_csv(sn_csv)
    z  = df["z"].to_numpy(float)
    mu = df["mu"].to_numpy(float)

    if sn_cov_path:
        M = load_spd_any(sn_cov_path)
        L, is_prec = cholesky_from_cov_or_prec(M, cov_kind, scale=scale)
    else:
        # fall back to diagonal from 'sigma'
        sig = df["sigma"].to_numpy(float)
        C = np.diag(sig**2)
        L, is_prec = cholesky_from_cov_or_prec(C, "cov", scale=1.0)
    return z, mu, L, is_prec

@dataclass
class BAOPack:
    z: np.ndarray
    y: np.ndarray
    L: np.ndarray
    is_prec: bool
    n: int

def load_bao(bao_csv: Optional[str], bao_cov: Optional[str]) -> Optional[BAOPack]:
    if not bao_csv:
        return None
    df = pd.read_csv(bao_csv)
    # expect columns: z, y, sigma (if no covariance given)
    z = df["z"].to_numpy(float)
    y = df["y"].to_numpy(float)
    if "sigma" in df.columns and not bao_cov:
        sig = df["sigma"].to_numpy(float)
        Cb = np.diag(sig**2)
        Lb, is_prec = cholesky_from_cov_or_prec(Cb, "cov", scale=1.0)
    else:
        Mb = load_spd_any(bao_cov) if bao_cov else None
        if Mb is None:
            raise ValueError("BAO covariance missing: provide --bao-cov or a 'sigma' column.")
        Lb, is_prec = cholesky_from_cov_or_prec(Mb, "cov", scale=1.0)
    return BAOPack(z=z, y=y, L=Lb, is_prec=is_prec, n=len(z))

# ------------------ cosmology pieces ---------------------

c_over_H0_100 = 2997.92458  # Mpc for H0 = 100 km/s/Mpc

def w_resonant(z, A, f, phi, gamma):
    z = np.asarray(z, float)
    return -1.0 + A * np.sin(f * np.log1p(z) + phi) * np.exp(-gamma * z)

def dark_energy_factor(z, A, f, phi, gamma, anchor_eps=0.0, anchor_zR=0.5, nz=400):
    """
    X_de(z) = exp[ 3 ∫_0^z (1+w(z'))/(1+z') dz' ] with phi -> phi_eff(z)
    """
    z = float(z)
    if z <= 0.0:
        return 1.0
    zz = np.linspace(0.0, z, max(8, int(nz)))
    phi_eff = phi + anchor_eps * np.exp(-(zz/anchor_zR)**2)
    w = w_resonant(zz, A, f, phi_eff, gamma)
    integrand = (1.0 + w) / (1.0 + zz)
    I = trapezoid(integrand, zz)
    return float(np.exp(3.0 * I))

def E_of_z(z, Om, A, f, phi, gamma, anchor_eps=0.0, anchor_zR=0.5, nz_int=400):
    Xde = dark_energy_factor(z, A, f, phi, gamma, anchor_eps=anchor_eps, anchor_zR=anchor_zR, nz=nz_int)
    return math.sqrt(Om*(1.0+z)**3 + (1.0-Om)*Xde)

def build_background_grid(zmax, H0, Om, A, f, phi, gamma, anchor_eps=0.0, anchor_zR=0.5, nz=800):
    z = np.linspace(0.0, float(zmax), max(16, int(nz)))
    Ez = np.array([E_of_z(t, Om, A, f, phi, gamma, anchor_eps=anchor_eps, anchor_zR=anchor_zR, nz_int=400) for t in z], float)
    invE = 1.0 / np.clip(Ez, 1e-12, None)
    # Comoving distance (flat): Dc = (c/H0) ∫ dz/E(z)
    Dc = (c_over_H0_100 / (H0/100.0)) * cumulative_trapezoid(invE, z, initial=0.0)
    Ez_at = lambda zz: np.interp(np.asarray(zz, float), z, Ez)
    Dc_at = lambda zz: np.interp(np.asarray(zz, float), z, Dc)
    return z, Dc, Dc_at, Ez_at

def mu_theory(z, H0, Om, A, f, phi, gamma, anchor_eps=0.0, anchor_zR=0.5):
    z = np.asarray(z, float)
    zmax = float(np.max(z)) + 0.05
    _, Dc, Dc_at, _ = build_background_grid(zmax, H0, Om, A, f, phi, gamma, anchor_eps=anchor_eps, anchor_zR=anchor_zR, nz=800)
    Dl = (1.0 + z) * Dc_at(z)
    return 5.0*np.log10(np.clip(Dl, 1e-20, None)) + 25.0

def bao_theory_vector(z_bao, bao_kind, H0, Om, rd, A, f, phi, gamma, anchor_eps=0.0, anchor_zR=0.5):
    z_bao = np.asarray(z_bao, float)
    zmax = float(np.max(z_bao)) + 0.05
    _, Dc, Dc_at, _ = build_background_grid(zmax, H0, Om, A, f, phi, gamma, anchor_eps=anchor_eps, anchor_zR=anchor_zR, nz=800)
    if bao_kind == "dm_over_rd":
        return Dc_at(z_bao) / rd
    elif bao_kind == "dm":
        return Dc_at(z_bao)
    else:
        raise ValueError("bao_kind must be 'dm_over_rd' or 'dm'")

# -------------------- chi^2 builders ---------------------

FULL_ORDER = ["A","f","phi","gamma","Om","rd","anchor_eps","dmu"]

def assemble_defaults(args) -> Dict[str, float]:
    return dict(
        A=float(args.A), f=float(args.f), phi=float(args.phi), gamma=float(args.gamma),
        Om=float(args.Om), rd=float(args.rd),
        anchor_eps=float(args.anchor_eps), dmu=float(args.dmu),
        H0=float(args.H0), anchor_zR=float(args.anchor_zR)
    )

def build_free_spec(args) -> Tuple[List[str], List[Tuple[float,float]], List[float]]:
    names: List[str] = []
    bounds: List[Tuple[float,float]] = []
    x0: List[float] = []

    def add(name: str, val: float, bnd: Tuple[float,float], is_free: bool):
        if is_free:
            names.append(name)
            bounds.append(bnd)
            x0.append(val)

    add("A",     float(args.A),     parse_bounds(args.A_bounds,     (0.0,1.0)),                      True)  # we always include A (can be pinned by bounds)
    add("f",     float(args.f),     parse_bounds(args.f_bounds,     (1.0,5.0)),                      True)
    add("phi",   float(args.phi),   parse_bounds(args.phi_bounds,   (-math.pi, math.pi)),            True)
    add("gamma", float(args.gamma), parse_bounds(args.gamma_bounds, (0.0,0.5)),                      True)
    add("Om",    float(args.Om),    parse_bounds(args.Om_bounds,    (0.05,0.70)),                    bool(args.free_Om))
    add("rd",    float(args.rd),    parse_bounds(args.rd_bounds,    (120.0,170.0)) if args.rd_bounds else (float(args.rd), float(args.rd)), bool(args.free_rd))
    add("anchor_eps", float(args.anchor_eps), parse_bounds(args.anchor_eps_bounds, (-0.6,0.6)),      bool(args.free_anchor))
    add("dmu",   float(args.dmu),   parse_bounds(args.dmu_bounds,   (-2.0,2.0)) if args.free_dmu else (float(args.dmu), float(args.dmu)),   bool(args.free_dmu))

    return names, bounds, x0

def theta_to_full(theta: List[float], names: List[str], defaults: Dict[str,float]) -> Dict[str,float]:
    params = {k: float(v) for k, v in defaults.items()}
    for k, v in zip(names, theta):
        params[k] = float(v)
    return params

def make_sn_chi2(z_sn, mu_sn, L_sn, sn_is_prec, H0, anchor_zR, prior_A_sigma: Optional[float]):
    z_sn = np.asarray(z_sn, float)
    mu_sn = np.asarray(mu_sn, float)

    def quad(res):
        if sn_is_prec:
            y = L_sn @ res
        else:
            y = solve(L_sn, res)
        return float(y @ y)

    def chi2(params: Dict[str,float]) -> float:
        mu_th = mu_theory(z_sn, H0, params["Om"], params["A"], params["f"], params["phi"], params["gamma"],
                          anchor_eps=params["anchor_eps"], anchor_zR=anchor_zR)
        # IMPORTANT: dmu ALWAYS applied
        resid = mu_sn - (mu_th + params["dmu"])
        c2 = quad(resid)
        if prior_A_sigma and prior_A_sigma > 0:
            c2 += (params["A"] / prior_A_sigma) ** 2
        return float(c2)

    return chi2

def make_bao_chi2(bao_pack: Optional[BAOPack], bao_kind: str, H0, anchor_zR):
    if bao_pack is None:
        return lambda params: 0.0

    z = bao_pack.z
    y = bao_pack.y
    L = bao_pack.L
    is_prec = bao_pack.is_prec

    def quad(res):
        if is_prec:
            v = L @ res
        else:
            v = solve(L, res)
        return float(v @ v)

    def chi2(params: Dict[str,float]) -> float:
        y_th = bao_theory_vector(z, bao_kind, H0, params["Om"], params["rd"], params["A"], params["f"], params["phi"], params["gamma"],
                                 anchor_eps=params["anchor_eps"], anchor_zR=anchor_zR)
        # dmu DOES NOT affect BAO
        return quad(y - y_th)

    return chi2

# ----------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--sn-csv", required=True)
    ap.add_argument("--sn-cov", required=True)
    ap.add_argument("--sn-cov-kind", choices=["cov","prec","auto"], default="cov")
    ap.add_argument("--sn-scale", type=float, default=1.0)
    ap.add_argument("--bao-csv", default=None)
    ap.add_argument("--bao-cov", default=None)
    ap.add_argument("--bao-kind", choices=["dm_over_rd","dm"], default="dm_over_rd")

    # Fiducials
    ap.add_argument("--fid-H0", dest="H0", type=float, default=70.0)
    ap.add_argument("--fid-Om", dest="Om", type=float, default=0.3)
    ap.add_argument("--fid-rd", dest="rd", type=float, default=147.1)

    # Resonant params (start)
    ap.add_argument("--A", type=float, default=0.0)
    ap.add_argument("--f", type=float, default=2.0)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)

    # Anchor tweak
    ap.add_argument("--anchor-eps", type=float, default=0.0)
    ap.add_argument("--anchor-zR",  type=float, default=0.5)

    # SN intercept
    ap.add_argument("--dmu", type=float, default=0.0)

    # Bounds
    ap.add_argument("--A-bounds", default="0.0,1.0")
    ap.add_argument("--f-bounds", default="1.0,5.0")
    ap.add_argument("--phi-bounds", default="-3.1415926536,3.1415926536")
    ap.add_argument("--gamma-bounds", default="0.0,0.5")
    ap.add_argument("--Om-bounds", default="0.05,0.70")
    ap.add_argument("--rd-bounds", default=None)
    ap.add_argument("--anchor-eps-bounds", default="-0.6,0.6")
    ap.add_argument("--dmu-bounds", default=None)

    # Free flags
    ap.add_argument("--free-Om", action="store_true")
    ap.add_argument("--free-rd", action="store_true")
    ap.add_argument("--free-anchor", action="store_true")
    ap.add_argument("--free-dmu", action="store_true")

    # Priors / fit control
    ap.add_argument("--prior-A-sigma", type=float, default=None)
    ap.add_argument("--verbose-every", type=int, default=25)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--bg-nz", type=int, default=800)
    ap.add_argument("--out", default="outputs/joint_with_anchor.json")

    args = ap.parse_args()

    # auto-detect sn_cov_kind if requested
    cov_kind = args.sn_cov_kind
    if cov_kind == "auto":
        cov_kind = "prec" if args.sn_cov.endswith(".spd.npy") else "cov"

    # Load data
    z_sn, mu_sn, L_sn, sn_is_prec = load_sn(args.sn_csv, args.sn_cov, cov_kind=cov_kind, scale=args.sn_scale)
    bao_pack = load_bao(args.bao_csv, args.bao_cov) if args.bao_csv else None

    # Defaults + free spec
    defaults = assemble_defaults(args)
    names, bounds, x0 = build_free_spec(args)

    # Closures
    sn_chi2  = make_sn_chi2(z_sn, mu_sn, L_sn, sn_is_prec, args.H0, args.anchor_zR, args.prior_A_sigma)
    bao_chi2 = make_bao_chi2(bao_pack, args.bao_kind, args.H0, args.anchor_zR)

    # Total chi2 for optimizer
    it_counter = {"n": 0}
    def total_chi2(theta):
        params = theta_to_full(theta, names, defaults)
        c2 = sn_chi2(params) + bao_chi2(params)
        return c2

    def cb(xk):
        it_counter["n"] += 1
        n = it_counter["n"]
        if args.verbose_every and n % args.verbose_every == 0:
            _progress(f"eval {n}: chi2={total_chi2(xk):.3f}")

    t0 = time.time()
    res = minimize(total_chi2, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=args.maxiter), callback=cb)
    elapsed = time.time() - t0

    # Best params (fill from defaults then update with result if success)
    best = theta_to_full(x0, names, defaults)
    if res.success:
        best = theta_to_full(list(res.x), names, defaults)

    # Evaluate breakdown at best
    c2_sn_best  = sn_chi2(best)
    c2_bao_best = bao_chi2(best)
    c2_tot      = c2_sn_best + c2_bao_best

    # Bounds hit info
    hit: Dict[str,bool] = {}
    for nm, (lo, hi) in zip(names, bounds):
        val = best[nm]
        hit[nm] = bool(abs(val - lo) < 1e-12 or abs(val - hi) < 1e-12)

    ndof = (len(z_sn) + (bao_pack.n if bao_pack else 0)) - len(names)

    out = {
        "best_params": {
            "H0": float(args.H0),
            "Om": float(best["Om"]),
            "rd": float(best["rd"]),
            "A": float(best["A"]),
            "f": float(best["f"]),
            "phi": float(best["phi"]),
            "gamma": float(best["gamma"]),
            "anchor_eps": float(best["anchor_eps"]),
            "anchor_zR": float(args.anchor_zR),
            "dmu": float(best["dmu"]),
        },
        "fit_params": names,
        "chi2": float(c2_tot),
        "ndof": int(ndof),
        "success": bool(res.success),
        "message": str(res.message) if hasattr(res, "message") else "",
        "restarts": 0,
        "breakdown": {
            "chi2_bao": float(c2_bao_best),
            "chi2_sn": float(c2_sn_best),
        },
        "timing_sec": float(elapsed),
        "sn_cov_kind_used": cov_kind,
        "sn_scale_used": float(args.sn_scale),
        "bao_kind_used": args.bao_kind,
        "bg_nz": int(args.bg_nz),
        "sn_points": int(len(z_sn)),
        "bao_points": int(bao_pack.n if bao_pack else 0),
        "hit_bounds": hit,
        "evals": int(it_counter["n"]),
        "config": {
            "bounds": {
                "A": args.A_bounds, "f": args.f_bounds, "phi": args.phi_bounds, "gamma": args.gamma_bounds,
                "Om": args.Om_bounds, "rd": args.rd_bounds, "anchor_eps": args.anchor_eps_bounds, "dmu": args.dmu_bounds
            },
            "free_flags": {
                "Om": bool(args.free_Om), "rd": bool(args.free_rd),
                "anchor": bool(args.free_anchor), "dmu": bool(args.free_dmu)
            }
        }
    }

    # Print + write
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
