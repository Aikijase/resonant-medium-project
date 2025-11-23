#!/usr/bin/env python3
"""
Joint BAO+SN with resonant modulation on BAO distances.

- SN χ²: uses joint_fit_resonant_rd.make_sn_chi2.
- BAO χ²: ΛCDM distances DM/rd and DH/rd, multiplied by
      M(z) = 1 + A*cos(f*ln(1+z)+phi)*exp(-gamma*z),
  using full covariance via Cholesky.

Options:
  --prior-A-sigma σA         add ((A-0)/σA)^2 to χ² if set
  --prior-gamma-sigma σγ     add ((γ-0)/σγ)^2 to χ² if set
  --fix-gamma-zero           fix γ = 0 via bounds

Run:
  PYTHONPATH=. python3 -u tools/res_joint_bao_resonant2.py \
    --bao-csv data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv \
    --bao-cov data/desi_dr1_bao/bao_covariance_plus_lya.csv \
    --sn-csv  data/pantheon_plus/sn_MATCHED_mu.csv \
    --sn-cov  data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.spd.csv \
    --out-prefix outputs/res_bao_res_v2 \
    --optimizer lbfgsb --max-evals 250
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
    p = argparse.ArgumentParser(description="Joint BAO+SN with resonant BAO modulation.")
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
    p.add_argument("--fix-gamma-zero", action="store_true")
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

# ---------- χ² builders ----------
def make_sn_fun(sn_pack, H0, Om):
    fn = J.make_sn_chi2
    sig = inspect.signature(fn)
    ctx = {**sn_pack, "sn_pack": sn_pack, "H0": H0, "Om": Om, "Om0": Om}
    kwargs = {k: v for k, v in ctx.items() if k in sig.parameters}
    return fn(**kwargs)

def make_bao_fun_resonant(bao_pack, H0, Om, rd):
    z, kind, y = bao_pack["z"], bao_pack["kind"], bao_pack["y"]
    L = bao_pack["L"]

    # precompute vanilla LCDM distances
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

# ---------- objective with priors ----------
def make_objective(sn_fun, bao_fun, prior_A_sigma=None, prior_gamma_sigma=None):
    def priors(theta):
        A, f, phi, gamma = theta[:4]
        pen = 0.0
        if prior_A_sigma and prior_A_sigma > 0:
            pen += (A / prior_A_sigma) ** 2
        if prior_gamma_sigma and prior_gamma_sigma > 0:
            pen += (gamma / prior_gamma_sigma) ** 2
        return pen

    def chi2(theta):
        return float(sn_fun(theta) + bao_fun(theta) + priors(theta))
    return chi2

# ---------- optimize ----------
def minimize(chi2, x0, method, max_evals, fix_gamma_zero=False):
    import scipy.optimize as opt
    if method == "lbfgsb":
        bounds = [(0.0, 5.0), (0.0, 8.0), (-math.pi, math.pi), (0.0, 2.0)]
        if fix_gamma_zero:
            bounds[-1] = (0.0, 0.0)  # lock gamma
        return opt.minimize(chi2, x0, method="L-BFGS-B",
                            bounds=bounds, options={"maxfun": max_evals})
    if method == "nelder":
        return opt.minimize(chi2, x0, method="Nelder-Mead",
                            options={"maxfev": max_evals})
    return opt.minimize(chi2, x0, method="Powell", options={"maxfev": max_evals})

# ---------- main ----------
def main():
    args = parse_args()

    sn_pack  = load_sn_pack(args.sn_csv,  args.sn_cov,  args.diag_only)
    bao_pack = load_bao_pack(args.bao_csv, args.bao_cov, args.diag_only)

    sn_fun  = make_sn_fun(sn_pack, args.H0, args.Om)
    bao_fun = make_bao_fun_resonant(bao_pack, args.H0, args.Om, args.rd)

    chi2 = make_objective(sn_fun, bao_fun,
                          prior_A_sigma=args.prior_A_sigma,
                          prior_gamma_sigma=args.prior_gamma_sigma)

    x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], float)
    t0 = time.time()
    res = minimize(chi2, x0, args.optimizer, args.max_evals, fix_gamma_zero=args.fix_gamma_zero)
    t1 = time.time()

    A,f,phi,gamma = map(float, res.x)
    out = {
        "best_params": {"A":A, "f":f, "phi":phi, "gamma":gamma,
                        "H0":args.H0, "Om":args.Om, "rd":args.rd},
        "chi2": float(res.fun),
        "ndof": int(sn_pack["C"].shape[0] + bao_pack["C"].shape[0] - 4),
        "success": bool(res.success),
        "message": str(res.message),
        "elapsed_s": t1 - t0,
        "nfev": int(getattr(res, "nfev", 0)),
        "breakdown": {"chi2_sn": float(sn_fun(res.x)),
                      "chi2_bao": float(bao_fun(res.x))}
    }

    p = Path(args.out_prefix).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {p}")

if __name__ == "__main__":
    main()
