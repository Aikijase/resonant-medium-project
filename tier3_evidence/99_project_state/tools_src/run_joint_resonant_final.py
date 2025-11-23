#!/usr/bin/env python3
"""
Joint BAO+SN with resonant BAO modulation and stable defaults.

Features
- SN χ² from joint_fit_resonant_rd.make_sn_chi2 (your working implementation)
- BAO χ² uses ΛCDM DM/rd and DH/rd multiplied by
      M(z) = 1 + A*cos(f*ln(1+z)+phi)*exp(-gamma*z)
  with full covariance via Cholesky
- Small-wiggle default bounds + optional Gaussian priors on A, gamma, f
- Utilities:
    --fix-gamma-zero     lock gamma = 0
    --bao-only           ignore SN (optimize BAO term + priors only)
    --auto-rescale-sn    rescale SN covariance numerically (match diag at θ0)

Run (typical):
  PYTHONPATH=. python3 -u tools/run_joint_resonant_final.py \
    --bao-csv data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv \
    --bao-cov data/desi_dr1_bao/bao_covariance_plus_lya.csv \
    --sn-csv  data/pantheon_plus/sn_MATCHED_mu.csv \
    --sn-cov  data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.spd.csv \
    --out-prefix outputs/res_final_v1 \
    --optimizer lbfgsb --max-evals 250 \
    --prior-A-sigma 0.6 --prior-gamma-sigma 0.3 --prior-f-mean 2.2 --prior-f-sigma 0.7
"""

from pathlib import Path
import sys, json, math, argparse, time, inspect
import numpy as np
import pandas as pd

# import project module (SN χ²)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fit_resonant_rd as J

C_LIGHT = 299792.458  # km/s

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Joint BAO+SN with resonant BAO and stable defaults.")
    p.add_argument("--bao-csv", required=True)
    p.add_argument("--bao-cov", required=True)
    p.add_argument("--sn-csv",  required=True)
    p.add_argument("--sn-cov",  required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--diag-only", action="store_true")

    p.add_argument("--H0", type=float, default=70.0)
    p.add_argument("--Om", type=float, default=0.3)
    p.add_argument("--rd", type=float, default=147.1)

    p.add_argument("--optimizer", default="lbfgsb", choices=["lbfgsb","nelder","powell"])
    p.add_argument("--max-evals", type=int, default=250)

    # initial guess for [A, f, phi, gamma]
    p.add_argument("--A0", type=float, default=0.4)
    p.add_argument("--f0", type=float, default=2.4)
    p.add_argument("--phi0", type=float, default=1.1)
    p.add_argument("--gamma0", type=float, default=0.0)

    # priors / constraints
    p.add_argument("--prior-A-sigma", type=float, default=None)
    p.add_argument("--prior-gamma-sigma", type=float, default=None)
    p.add_argument("--prior-f-mean", type=float, default=None)
    p.add_argument("--prior-f-sigma", type=float, default=None)
    p.add_argument("--fix-gamma-zero", action="store_true")

    # utilities
    p.add_argument("--bao-only", action="store_true")
    p.add_argument("--auto-rescale-sn", action="store_true",
                   help="Numerically rescale SN C and its factor to match diag-only χ² at θ0 (readability).")
    return p.parse_args()

# ---------- cosmology ----------
def Ez(z, Om, Ode):
    return np.sqrt(Om*(1+z)**3 + Ode)  # flat

def DH_over_rd(z, H0, Om, rd):
    Ode = 1.0 - Om
    return (C_LIGHT / H0) / rd / Ez(z, Om, Ode)

def DM_over_rd(z, H0, Om, rd, nz=800):
    z = np.atleast_1d(z).astype(float)
    Ode = 1.0 - Om
    out = np.empty_like(z, dtype=float)
    for i, zi in enumerate(z):
        if zi <= 0.0:
            out[i] = 0.0
            continue
        grid = np.linspace(0.0, zi, nz)
        E = Ez(grid, Om, Ode)
        h = grid[1] - grid[0]
        s = E[0]**-1 + E[-1]**-1 + 4.0*np.sum((E[1:-1:2])**-1) + 2.0*np.sum((E[2:-2:2])**-1)
        chi = (h/3.0) * s
        DM = (C_LIGHT / H0) * chi
        out[i] = DM / rd
    return out

def M_res(z, A, f, phi, gamma):
    delta = A * np.cos(f*np.log1p(z) + phi) * np.exp(-gamma*z)
    # tiny floor to stay positive
    return np.maximum(1.0 + delta, 1e-9)

# ---------- loaders ----------
def load_sn_pack(sn_csv: str, sn_cov_csv: str, diag_only: bool):
    df = pd.read_csv(sn_csv)
    z  = df["z"].to_numpy(float)  if "z"  in df.columns else df.iloc[:,0].to_numpy(float)
    mu = df["mu"].to_numpy(float) if "mu" in df.columns else df.iloc[:,1].to_numpy(float)
    C = pd.read_csv(sn_cov_csv, header=None).to_numpy(float)
    if diag_only:
        C = np.diag(np.diag(C))
    L = np.linalg.cholesky(C)
    return {
        "z": z, "mu": mu, "y": mu,
        "C": C, "cov": C,
        "L": L, "R": L, "L_or_R": L, "F": L,
        "is_prec": False
    }

def load_bao_pack(bao_csv: str, bao_cov_csv: str, diag_only: bool):
    df = pd.read_csv(bao_csv)
    if not {"kind","z"}.issubset(df.columns):
        raise ValueError("BAO CSV must include 'kind' and 'z'.")
    y = df["y_data"].to_numpy(float) if "y_data" in df.columns else \
        df.select_dtypes(include=[np.number]).iloc[:,-1].to_numpy(float)
    z = df["z"].to_numpy(float)
    kind = df["kind"].astype(str).to_numpy()
    C = pd.read_csv(bao_cov_csv, header=None).to_numpy(float)
    if diag_only:
        C = np.diag(np.diag(C))
    if C.shape[0] != y.shape[0]:
        raise ValueError(f"BAO cov shape {C.shape} != BAO vector length {y.shape[0]}")
    L = np.linalg.cholesky(C)
    return {"y": y, "z": z, "kind": kind, "C": C, "L": L}

# ---------- SN helpers ----------
def _make_kwargs(fn, context):
    sig = inspect.signature(fn)
    return {k: v for k, v in context.items() if k in sig.parameters}

def make_sn_fun(sn_pack, H0, Om):
    fn = J.make_sn_chi2
    ctx = {**sn_pack, "sn_pack": sn_pack, "H0": H0, "Om": Om, "Om0": Om}
    return fn(**_make_kwargs(fn, ctx))

def rescale_sn_pack_to_diag(sn_pack, H0, Om, theta0):
    """Numerically rescale SN C (and L) so full-cov χ² at θ0 equals diag-only χ² at θ0."""
    # full
    full_fun = make_sn_fun(sn_pack, H0, Om)
    chi2_full = float(full_fun(theta0))
    # diag baseline
    pack_d = dict(sn_pack)
    C = pack_d["C"]
    pack_d["C"] = np.diag(np.diag(C))
    Ld = np.linalg.cholesky(pack_d["C"])
    for k in ("L","R","L_or_R","F"): pack_d[k] = Ld
    diag_fun = make_sn_fun(pack_d, H0, Om)
    chi2_diag = float(diag_fun(theta0))
    if chi2_full > 0 and chi2_diag > 0:
        s = chi2_full / chi2_diag
        if np.isfinite(s) and s > 0:
            # C' = C / s, L' = L / sqrt(s)
            sn_pack["C"] = sn_pack["C"] / s
            sn_pack["L"] = sn_pack["L"] / math.sqrt(s)
            for k in ("R","L_or_R","F"): sn_pack[k] = sn_pack["L"]

# ---------- BAO χ² (resonant) ----------
def make_bao_fun_resonant(bao_pack, H0, Om, rd):
    z, kind, y = bao_pack["z"], bao_pack["kind"], bao_pack["y"]
    L = bao_pack["L"]
    dm = DM_over_rd(z, H0, Om, rd)
    dh = DH_over_rd(z, H0, Om, rd)
    mDM = (kind == "DM")
    mDH = (kind == "DH")

    def predict(theta):
        A, f, phi, gamma = map(float, theta[:4])
        mod = M_res(z, A, f, phi, gamma)
        y_th = np.empty_like(y, dtype=float)
        if mDM.any(): y_th[mDM] = mod[mDM] * dm[mDM]
        if mDH.any(): y_th[mDH] = mod[mDH] * dh[mDH]
        return y_th

    def chi2(theta):
        res = y - predict(theta)
        x = np.linalg.solve(L, res)  # L L^T = C
        return float(x @ x)

    return chi2

# ---------- Objective w/ priors ----------
def make_objective(sn_fun, bao_fun, prior_A_sigma=None, prior_gamma_sigma=None,
                   prior_f_mean=None, prior_f_sigma=None):
    def priors(theta):
        A, f, phi, gamma = theta[:4]
        pen = 0.0
        if prior_A_sigma and prior_A_sigma > 0:
            pen += (A / prior_A_sigma) ** 2
        if prior_gamma_sigma and prior_gamma_sigma > 0:
            pen += (gamma / prior_gamma_sigma) ** 2
        if (prior_f_mean is not None) and (prior_f_sigma and prior_f_sigma > 0):
            pen += ((f - prior_f_mean) / prior_f_sigma) ** 2
        return pen
    def chi2(theta):
        base = 0.0
        if sn_fun is not None:
            base += sn_fun(theta)
        if bao_fun is not None:
            base += bao_fun(theta)
        return float(base + priors(theta))
    return chi2

# ---------- Optimize ----------
def minimize(chi2, x0, method, max_evals, fix_gamma_zero=False):
    import scipy.optimize as opt
    if method == "lbfgsb":
        # Small-wiggle box to avoid runaway fits
        bounds = [(0.0, 1.5),     # A
                  (0.8, 4.0),     # f
                  (-math.pi, math.pi),
                  (0.0, 0.6)]     # gamma
        if fix_gamma_zero:
            bounds[-1] = (0.0, 0.0)
        return opt.minimize(chi2, x0, method="L-BFGS-B",
                            bounds=bounds, options={"maxfun": max_evals})
    if method == "nelder":
        return opt.minimize(chi2, x0, method="Nelder-Mead",
                            options={"maxfev": max_evals})
    return opt.minimize(chi2, x0, method="Powell", options={"maxfev": max_evals})

# ---------- Main ----------
def main():
    args = parse_args()

    # Load packs
    sn_pack  = load_sn_pack(args.sn_csv,  args.sn_cov,  args.diag_only)
    bao_pack = load_bao_pack(args.bao_csv, args.bao_cov, args.diag_only)

    # Optional numeric rescale for SN (readability)
    x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], float)
    if (not args.bao_only) and args.auto_rescale_sn and (not args.diag_only):
        rescale_sn_pack_to_diag(sn_pack, args.H0, args.Om, x0)

    # Build χ² terms
    sn_fun  = None if args.bao_only else make_sn_fun(sn_pack, args.H0, args.Om)
    bao_fun = make_bao_fun_resonant(bao_pack, args.H0, args.Om, args.rd)

    chi2 = make_objective(
        sn_fun, bao_fun,
        prior_A_sigma=args.prior_A_sigma,
        prior_gamma_sigma=args.prior_gamma_sigma,
        prior_f_mean=args.prior_f_mean,
        prior_f_sigma=args.prior_f_sigma
    )

    # Optimize
    t0 = time.time()
    res = minimize(chi2, x0, args.optimizer, args.max_evals, fix_gamma_zero=args.fix_gamma_zero)
    t1 = time.time()

    A,f,phi,gamma = map(float, res.x)
    # Breakdown
    br_sn  = float(sn_fun(res.x))  if sn_fun  is not None else 0.0
    br_bao = float(bao_fun(res.x)) if bao_fun is not None else 0.0

    out = {
        "best_params": {"A":A, "f":f, "phi":phi, "gamma":gamma,
                        "H0":args.H0, "Om":args.Om, "rd":args.rd},
        "chi2": float(res.fun),
        "ndof": int(sn_pack["C"].shape[0] * (0 if args.bao_only else 1)
                    + bao_pack["C"].shape[0] - 4),
        "success": bool(res.success),
        "message": str(res.message),
        "elapsed_s": t1 - t0,
        "nfev": int(getattr(res, "nfev", 0)),
        "breakdown": {"chi2_sn": br_sn, "chi2_bao": br_bao}
    }

    p = Path(args.out_prefix).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {p}")

if __name__ == "__main__":
    main()
