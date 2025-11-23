#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interacting Dark Sector (IDS) background fitter for SN + BAO

Model:
  - Effective fluids: x_chi = rho_chi/rho_c0 (DM-like), x_phi = rho_phi/rho_c0 (DE-like)
  - Constant DE equation of state w_de (default -1)  [w_de = -1 recovers vacuum]
  - Energy exchange (horizon-permeated "rhythm"):
        Q = [eps_H * H * rho_chi + xi_H * H * rho_phi] * exp[-(z/zR)^2]
    In dimensionless densities (prime = d/d ln a):
        x_chi' = -3 x_chi + q(a)
        x_phi' = -3 (1 + w_de) x_phi - q(a)
      where q(a) = [eps_H * x_chi + xi_H * x_phi] * exp[-(z/zR)^2]
  - Flat FRW (k=0). Optional fixed Omega_b0, Omega_r0 (default 0).

Observables:
  - E(z) = H(z)/H0 from x_chi(a), x_phi(a), Omega_b0, Omega_r0
  - D_M(z) = (c/H0) ∫ dz / E(z)
  - SN: mu(z) = 5 log10[(1+z) D_M(z)] + 25 + dmu
  - BAO: DM(z) or DM(z)/r_d

CLI mirrors your previous scripts. JSON output includes breakdowns and bounds.
"""

from __future__ import annotations
import argparse, json, math, os, sys, time
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from numpy.linalg import cholesky, solve
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import minimize

# ----------------------------- utils -----------------------------

c_km_s = 299792.458
MPC_PER_M = 3.2407792896664e-23  # not needed; we work in Mpc with c/H0 trick
c_over_H0_100 = 2997.92458  # Mpc for H0 = 100 km/s/Mpc

def _ts(msg: str) -> str:
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"

def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5*(M + M.T)

def load_spd_any(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    if path.endswith(".npy"):
        return np.load(path)
    return pd.read_csv(path, header=None).to_numpy(dtype=float)

def cholesky_from_cov_or_prec(M: Optional[np.ndarray], kind: str, scale: float = 1.0):
    """Return (L, is_prec). If kind='cov': C = L L^T; if 'prec': P = L^T L."""
    if M is None:
        return None, False
    M = np.array(M, float)
    if scale is not None and scale != 1.0:
        M = float(scale) * M
    if kind == "cov":
        C = _sym(M) + 1e-12*np.eye(M.shape[0])
        return cholesky(C), False
    elif kind == "prec":
        P = _sym(M) + 1e-12*np.eye(M.shape[0])
        return cholesky(P).T, True
    else:
        raise ValueError("kind must be 'cov' or 'prec'")

def parse_bounds(s: Optional[str], default: Tuple[float,float]) -> Tuple[float,float]:
    if s is None: return default
    a,b = [float(t) for t in s.split(",")]
    return (a,b)

# ------------------------- data loaders --------------------------

def load_sn(sn_csv: str, sn_cov: Optional[str], cov_kind: str, scale: float):
    df = pd.read_csv(sn_csv)
    z  = df["z"].to_numpy(float)
    mu = df["mu"].to_numpy(float)
    if sn_cov:
        M = load_spd_any(sn_cov)
        L, is_prec = cholesky_from_cov_or_prec(M, cov_kind, scale=scale)
    else:
        sig = df["sigma"].to_numpy(float)
        C = np.diag(sig**2)
        L, is_prec = cholesky_from_cov_or_prec(C, "cov")
    return z, mu, L, is_prec

def load_bao_dm_or_dm_over_rd(bao_csv: Optional[str], bao_cov: Optional[str]):
    if not bao_csv:
        return None
    df = pd.read_csv(bao_csv)
    z = df["z"].to_numpy(float)
    y = df["y"].to_numpy(float)
    if bao_cov:
        Mb = load_spd_any(bao_cov)
        Lb, is_prec = cholesky_from_cov_or_prec(Mb, "cov")
    else:
        sig = df["sigma"].to_numpy(float)
        Cb = np.diag(sig**2)
        Lb, is_prec = cholesky_from_cov_or_prec(Cb, "cov")
    return {"z": z, "y": y, "L": Lb, "is_prec": is_prec, "n": len(z)}

# ---------------------- interacting background -------------------

def window_gauss_z(z: np.ndarray, zR: float) -> np.ndarray:
    if zR <= 0:  # no window
        return np.ones_like(z)
    return np.exp(-(z/zR)**2)

def integrate_ids_background(zmax: float,
                             Om0: float,
                             w_de: float,
                             epsH: float,
                             xiH: float,
                             zR: float,
                             Ob0: float,
                             Or0: float,
                             n_steps: int = 4000):
    """
    Integrate x_chi(a), x_phi(a) from a=1 down to a_min=1/(1+zmax).
    State in ln a with Heun's method (improved Euler). Radiation/baryons are fixed power-laws.
    Returns z_grid (ascending), E_grid, Dc_grid (Mpc/H0 units i.e. c/H0=2997.9 for H0=100)
    """
    a_min = 1.0/(1.0 + float(zmax))
    ln_a0, ln_amin = 0.0, math.log(a_min)
    N = max(1000, int(n_steps))
    dln = (ln_amin - ln_a0)/N  # negative step

    # initial conditions at a=1
    x_chi = float(Om0)             # treat all "matter" as the interacting component for background
    x_phi = max(1.0 - Om0 - Ob0 - Or0, 1e-12)  # flatness

    ln_a_list = [ln_a0]
    a_list = [1.0]
    z_list = [0.0]
    xchi_list = [x_chi]
    xphi_list = [x_phi]
    E2_list = [Or0 + Ob0 + x_chi + x_phi]  # at a=1

    for i in range(N):
        ln_a = ln_a_list[-1]
        a = math.exp(ln_a)
        z = 1.0/a - 1.0
        W = math.exp(- (z/zR)**2 ) if zR > 0 else 1.0
        q1 = (epsH * x_chi + xiH * x_phi) * W
        k1_chi = -3.0*x_chi + q1
        k1_phi = -3.0*(1.0 + w_de)*x_phi - q1

        # predictor
        x_chi_p = max(x_chi + dln*k1_chi, 1e-18)
        x_phi_p = max(x_phi + dln*k1_phi, 1e-18)
        a_p = math.exp(ln_a + dln)
        z_p = 1.0/a_p - 1.0
        Wp = math.exp(- (z_p/zR)**2 ) if zR > 0 else 1.0
        q2 = (epsH * x_chi_p + xiH * x_phi_p) * Wp
        k2_chi = -3.0*x_chi_p + q2
        k2_phi = -3.0*(1.0 + w_de)*x_phi_p - q2

        # corrector
        x_chi = max(x_chi + 0.5*dln*(k1_chi + k2_chi), 1e-18)
        x_phi = max(x_phi + 0.5*dln*(k1_phi + k2_phi), 1e-18)

        ln_a_new = ln_a + dln
        a_new = math.exp(ln_a_new)
        z_new = 1.0/a_new - 1.0
        E2_new = Or0*(a_new**-4) + Ob0*(a_new**-3) + x_chi + x_phi

        ln_a_list.append(ln_a_new)
        a_list.append(a_new)
        z_list.append(z_new)
        xchi_list.append(x_chi)
        xphi_list.append(x_phi)
        E2_list.append(max(E2_new, 1e-14))

    # ascending z
    z_grid = np.array(z_list[::-1])
    E_grid = np.sqrt(np.array(E2_list[::-1], float))
    # Precompute comoving distance integral in z
    invE = 1.0 / np.clip(E_grid, 1e-12, None)
    # z grid spacing is not uniform; trapezoid is fine
    Dc_over_cH0 = cumulative_trapezoid(invE, z_grid, initial=0.0)  # dimensionless ∫dz/E
    # Convert to Mpc: Dc = (c/H0)*∫dz/E = (c_over_H0_100 / (H0/100)) * integral
    return z_grid, E_grid, Dc_over_cH0

# --------------------------- predictions -------------------------

def build_background_and_interps(zmax, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, n_steps=4000):
    z_bg, E_bg, I_bg = integrate_ids_background(zmax, Om0, w_de, epsH, xiH, zR, Ob0, Or0, n_steps=n_steps)
    # Interpolants
    def E_at(z):
        z = np.asarray(z, float)
        return np.interp(z, z_bg, E_bg)
    def Dc_at(z):
        z = np.asarray(z, float)
        I = np.interp(z, z_bg, I_bg)
        return (c_over_H0_100 / (H0/100.0)) * I
    return z_bg, E_at, Dc_at

def mu_theory(z, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, n_steps=4000):
    z = np.asarray(z, float)
    zmax = float(np.max(z)) + 0.05
    _, _, Dc_at = build_background_and_interps(zmax, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, n_steps=n_steps)
    Dl = (1.0 + z) * Dc_at(z)
    return 5.0*np.log10(np.clip(Dl, 1e-20, None)) + 25.0

def bao_DM_or_DM_over_rd(z_bao, bao_kind, H0, Om0, rd, w_de, epsH, xiH, zR, Ob0, Or0, n_steps=4000):
    z_bao = np.asarray(z_bao, float)
    zmax = float(np.max(z_bao)) + 0.05
    _, _, Dc_at = build_background_and_interps(zmax, H0, Om0, w_de, epsH, xiH, zR, Ob0, Or0, n_steps=n_steps)
    DM = Dc_at(z_bao)
    if bao_kind == "dm_over_rd":
        return DM / rd
    elif bao_kind == "dm":
        return DM
    else:
        raise ValueError("bao_kind must be 'dm_over_rd' or 'dm'")

# ------------------------------ chi2 ------------------------------

def make_sn_chi2(z, mu, L, is_prec, H0, Ob0, Or0, n_steps):
    z = np.asarray(z, float); mu = np.asarray(mu, float)

    def quad(res):
        y = L @ res if is_prec else solve(L, res)
        return float(y @ y)

    def chi2(params: Dict[str, float]) -> float:
        mu_th = mu_theory(z, H0, params["Om0"], params["w_de"], params["epsH"], params["xiH"], params["zR"], Ob0, Or0, n_steps=n_steps)
        resid = mu - (mu_th + params["dmu"])  # ALWAYS apply dmu
        return quad(resid)
    return chi2

def make_bao_chi2(bao_pack, bao_kind, H0, Ob0, Or0, n_steps):
    if bao_pack is None:
        return lambda params: 0.0
    z = bao_pack["z"]; y = bao_pack["y"]; L = bao_pack["L"]; is_prec = bao_pack["is_prec"]

    def quad(res):
        v = L @ res if is_prec else solve(L, res)
        return float(v @ v)

    def chi2(params: Dict[str, float]) -> float:
        y_th = bao_DM_or_DM_over_rd(z, bao_kind, H0, params["Om0"], params["rd"], params["w_de"], params["epsH"], params["xiH"], params["zR"], Ob0, Or0, n_steps=n_steps)
        return quad(y - y_th)
    return chi2

# ------------------------------ main ------------------------------

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

    # Background reference
    ap.add_argument("--fid-H0", dest="H0", type=float, default=70.0)

    # Present-day fractions (flat)
    ap.add_argument("--fid-Om", dest="Om0", type=float, default=0.3)
    ap.add_argument("--Omega_b0", type=float, default=0.0)
    ap.add_argument("--Omega_r0", type=float, default=0.0)

    # IDS params (start)
    ap.add_argument("--w_de", type=float, default=-1.0)
    ap.add_argument("--epsH", type=float, default=0.0)
    ap.add_argument("--xiH",  type=float, default=0.0)
    ap.add_argument("--zR",   type=float, default=0.6)
    ap.add_argument("--rd",   type=float, default=147.1)
    ap.add_argument("--dmu",  type=float, default=0.0)

    # Bounds
    ap.add_argument("--Om-bounds", default="0.05,0.70")
    ap.add_argument("--wde-bounds", default="-1.2,-0.8")
    ap.add_argument("--epsH-bounds", default="-0.05,0.05")
    ap.add_argument("--xiH-bounds",  default="-0.05,0.05")
    ap.add_argument("--zR-bounds",   default="0.2,1.2")
    ap.add_argument("--rd-bounds",   default=None)
    ap.add_argument("--dmu-bounds",  default=None)

    # Free flags
    ap.add_argument("--free-Om", action="store_true")
    ap.add_argument("--free-wde", action="store_true")
    ap.add_argument("--free-epsH", action="store_true")
    ap.add_argument("--free-xiH", action="store_true")
    ap.add_argument("--free-zR", action="store_true")
    ap.add_argument("--free-rd", action="store_true")
    ap.add_argument("--free-dmu", action="store_true")

    # Controls
    ap.add_argument("--bg-steps", type=int, default=4000, help="ln a steps for background integrator")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--verbose-every", type=int, default=25)
    ap.add_argument("--out", default="outputs/ids_fit.json")

    ap.add_argument("--sn-scale-note", default="", help="optional note; echoed in JSON")

    args = ap.parse_args()

    # sn kind
    cov_kind = args.sn_cov_kind
    if cov_kind == "auto":
        cov_kind = "prec" if args.sn_cov.endswith(".spd.npy") else "cov"

    # Load data
    z_sn, mu_sn, L_sn, sn_is_prec = load_sn(args.sn_csv, args.sn_cov, cov_kind=cov_kind, scale=args.sn_scale)
    bao_pack = load_bao_dm_or_dm_over_rd(args.bao_csv, args.bao_cov) if args.bao_csv else None

    # Build chi2 closures
    sn_chi2  = make_sn_chi2(z_sn, mu_sn, L_sn, sn_is_prec, args.H0, args.Omega_b0, args.Omega_r0, n_steps=args.bg_steps)
    bao_chi2 = make_bao_chi2(bao_pack, args.bao_kind, args.H0, args.Omega_b0, args.Omega_r0, n_steps=args.bg_steps)

    # Defaults and free vector
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
    if args.free_dmu:
        dmu_b = parse_bounds(args.dmu_bounds, (-2.0,2.0)) if args.dmu_bounds else (-2.0,2.0)
        add("dmu", defaults["dmu"], dmu_b, True)

    # helper to merge theta into full params
    def theta_to_params(theta):
        params = dict(defaults)
        for k,v in zip(names, theta):
            params[k] = float(v)
        return params

    it_counter = {"n": 0}
    def total_chi2(theta):
        p = theta_to_params(theta)
        return sn_chi2(p) + bao_chi2(p)

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

    # Evaluate breakdown
    c2_sn = sn_chi2(best)
    c2_bao = bao_chi2(best)
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
            "dmu":  float(best["dmu"]),
            "Ob0":  float(best["Ob0"]),
            "Or0":  float(best["Or0"]),
        },
        "fit_params": names,
        "chi2": float(c2_tot),
        "ndof": int(ndof),
        "success": bool(res.success),
        "message": str(res.message) if hasattr(res, "message") else "",
        "breakdown": {"chi2_bao": float(c2_bao), "chi2_sn": float(c2_sn)},
        "timing_sec": float(elapsed),
        "sn_cov_kind_used": cov_kind,
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
                "zR": args.zR_bounds, "rd": args.rd_bounds, "dmu": args.dmu_bounds
            },
            "free": {
                "Om0": bool(args.free_Om), "w_de": bool(args.free_wde),
                "epsH": bool(args.free_epsH), "xiH": bool(args.free_xiH),
                "zR": bool(args.free_zR), "rd": bool(args.free_rd), "dmu": bool(args.free_dmu)
            }
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
