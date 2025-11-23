#!/usr/bin/env python3
"""
Joint BAO+SN with resonant BAO modulation + global BAO scale and per-kind tweaks.

Scales:
  s_all  (global, wide bounds) and small deltas δ_DM, δ_DH (tight)
  s_DM = s_all * (1 + δ_DM)
  s_DH = s_all * (1 + δ_DH)

Why: If both DM/DH residual means are positive, a single global uplift is cleaner
than shoving both s_DM and s_DH to their caps.

Params (in order when fit_scales):
  θ = [A, f, phi, gamma, s_all, δ_DM, δ_DH]

Recommended priors:
  A ~ N(0, σ_A=0.5),  γ ~ N(0, σ_γ=0.25),
  f ~ N(μ_f=3.5, σ_f=0.8),
  s_all ~ N(1, σ=0.10),  δ_* ~ N(0, σ=0.04).

Bounds:
  A∈[0,3], f∈[1,6], phi∈[-π,π], γ∈[0,1.2]
  s_all∈[0.6,1.6], δ_*∈[-0.25,0.25]
"""

from pathlib import Path
import sys, json, math, argparse, time, inspect
import numpy as np
import pandas as pd

# import SN χ² from your repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fit_resonant_rd as J

C_LIGHT = 299792.458  # km/s

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Joint BAO+SN with resonant BAO + global scale + per-kind tweaks.")
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
    p.add_argument("--max-evals", type=int, default=350)

    # initial guess for [A, f, phi, gamma]
    p.add_argument("--A0", type=float, default=0.4)
    p.add_argument("--f0", type=float, default=2.4)
    p.add_argument("--phi0", type=float, default=1.1)
    p.add_argument("--gamma0", type=float, default=0.0)

    # fit scales?
    p.add_argument("--fit-bao-scales", action="store_true",
                   help="Fit global BAO scale (s_all) and small per-kind deltas (delta_DM, delta_DH).")

    # priors on resonant params
    p.add_argument("--prior-A-sigma", type=float, default=0.5)
    p.add_argument("--prior-gamma-sigma", type=float, default=0.25)
    p.add_argument("--prior-f-mean", type=float, default=3.5)
    p.add_argument("--prior-f-sigma", type=float, default=0.8)

    # priors on scales
    p.add_argument("--prior-sall-sigma", type=float, default=0.10)   # ~10% on global BAO scale
    p.add_argument("--prior-delta-sigma", type=float, default=0.04)  # small per-kind tweaks ~4%

    # utilities
    p.add_argument("--fix-gamma-zero", action="store_true")
    p.add_argument("--bao-only", action="store_true")
    p.add_argument("--auto-rescale-sn", action="store_true")
    p.add_argument("--dump-residuals", action="store_true")
    return p.parse_args()

# ---------- cosmology ----------
def Ez(z, Om, Ode): return np.sqrt(Om*(1+z)**3 + Ode)
def DH_over_rd(z, H0, Om, rd):
    Ode = 1.0 - Om
    return (C_LIGHT / H0) / rd / Ez(z, Om, Ode)
def DM_over_rd(z, H0, Om, rd, nz=800):
    z = np.atleast_1d(z).astype(float)
    Ode = 1.0 - Om
    out = np.empty_like(z, dtype=float)
    for i, zi in enumerate(z):
        if zi <= 0: out[i]=0.0; continue
        g = np.linspace(0.0, zi, nz)
        E = Ez(g, Om, Ode)
        h = g[1]-g[0]
        s = E[0]**-1 + E[-1]**-1 + 4*np.sum((E[1:-1:2])**-1) + 2*np.sum((E[2:-2:2])**-1)
        DM = (h/3.0)*s*(C_LIGHT/H0)
        out[i] = DM/rd
    return out
def M_res(z, A, f, phi, gamma):
    delta = A*np.cos(f*np.log1p(z) + phi)*np.exp(-gamma*z)
    return np.maximum(1.0+delta, 1e-9)

# ---------- loaders ----------
def load_sn_pack(sn_csv, sn_cov_csv, diag_only):
    df = pd.read_csv(sn_csv)
    z  = df["z"].to_numpy(float)  if "z" in df.columns else df.iloc[:,0].to_numpy(float)
    mu = df["mu"].to_numpy(float) if "mu" in df.columns else df.iloc[:,1].to_numpy(float)
    C  = pd.read_csv(sn_cov_csv, header=None).to_numpy(float)
    if diag_only: C = np.diag(np.diag(C))
    L  = np.linalg.cholesky(C)
    return {"z":z, "mu":mu, "y":mu, "C":C, "L":L, "R":L, "L_or_R":L, "F":L, "is_prec":False}

def load_bao_pack(bao_csv, bao_cov_csv, diag_only):
    df = pd.read_csv(bao_csv)
    if not {"kind","z"}.issubset(df.columns):
        raise ValueError("BAO CSV must include 'kind' and 'z'")
    y    = df["y_data"].to_numpy(float) if "y_data" in df.columns else \
           df.select_dtypes(include=[np.number]).iloc[:,-1].to_numpy(float)
    z    = df["z"].to_numpy(float)
    kind = df["kind"].astype(str).to_numpy()
    C    = pd.read_csv(bao_cov_csv, header=None).to_numpy(float)
    if diag_only: C = np.diag(np.diag(C))
    if C.shape[0] != y.shape[0]:
        raise ValueError(f"BAO cov shape {C.shape} != len(y) {y.shape[0]}")
    L = np.linalg.cholesky(C)
    return {"y":y, "z":z, "kind":kind, "C":C, "L":L}

# ---------- SN helpers ----------
def _kw_for(fn, ctx):
    import inspect as _ins
    sig = _ins.signature(fn)
    return {k:v for k,v in ctx.items() if k in sig.parameters}
def make_sn_fun(pack, H0, Om):
    fn = J.make_sn_chi2
    ctx = {**pack, "sn_pack":pack, "H0":H0, "Om":Om, "Om0":Om}
    return fn(**_kw_for(fn, ctx))
def rescale_sn_pack_to_diag(sn_pack, H0, Om, theta0):
    full_fun = make_sn_fun(sn_pack, H0, Om)
    chi2_full = float(full_fun(theta0))
    C = sn_pack["C"]
    Cdiag = np.diag(np.diag(C))
    Ld = np.linalg.cholesky(Cdiag)
    tmp = {"z":sn_pack["z"], "mu":sn_pack["mu"], "y":sn_pack["y"], "C":Cdiag,
           "L":Ld, "R":Ld, "L_or_R":Ld, "F":Ld, "is_prec":False}
    chi2_diag = float(make_sn_fun(tmp, H0, Om)(theta0))
    if chi2_full>0 and chi2_diag>0:
        s = chi2_full/chi2_diag
        if np.isfinite(s) and s>0:
            sn_pack["C"] /= s
            sn_pack["L"] /= math.sqrt(s)
            for k in ("R","L_or_R","F"): sn_pack[k] = sn_pack["L"]

# ---------- BAO χ² with global scale ----------
def make_bao_fun_resonant(bao_pack, H0, Om, rd, fit_scales=False):
    z, kind, y = bao_pack["z"], bao_pack["kind"], bao_pack["y"]
    L = bao_pack["L"]
    dm = DM_over_rd(z, H0, Om, rd)
    dh = DH_over_rd(z, H0, Om, rd)
    mDM = (kind=="DM")
    mDH = (kind=="DH")
    def predict(theta):
        A,f,phi,gamma = map(float, theta[:4])
        if fit_scales:
            s_all = float(theta[4]); dDM = float(theta[5]); dDH = float(theta[6])
            sDM = s_all*(1.0 + dDM)
            sDH = s_all*(1.0 + dDH)
        else:
            sDM = sDH = 1.0
        mod = M_res(z, A, f, phi, gamma)
        yth = np.empty_like(y, float)
        if mDM.any(): yth[mDM] = sDM*mod[mDM]*dm[mDM]
        if mDH.any(): yth[mDH] = sDH*mod[mDH]*dh[mDH]
        return yth
    def chi2(theta):
        res = y - predict(theta)
        x = np.linalg.solve(L, res)
        return float(x@x)
    return chi2

# ---------- objective + priors ----------
def make_objective(sn_fun, bao_fun,
                   prior_A_sigma=0.5, prior_gamma_sigma=0.25,
                   prior_f_mean=3.5, prior_f_sigma=0.8,
                   fit_scales=False, prior_sall_sigma=0.10, prior_delta_sigma=0.04):
    def priors(theta):
        A,f,phi,gamma = theta[:4]
        pen = 0.0
        if prior_A_sigma>0:      pen += (A/prior_A_sigma)**2
        if prior_gamma_sigma>0:  pen += (gamma/prior_gamma_sigma)**2
        if prior_f_sigma and prior_f_sigma>0 and prior_f_mean is not None:
            pen += ((f-prior_f_mean)/prior_f_sigma)**2
        if fit_scales:
            s_all = float(theta[4]); dDM = float(theta[5]); dDH = float(theta[6])
            if prior_sall_sigma and prior_sall_sigma>0:
                pen += ((s_all-1.0)/prior_sall_sigma)**2
            if prior_delta_sigma and prior_delta_sigma>0:
                pen += (dDM/prior_delta_sigma)**2 + (dDH/prior_delta_sigma)**2
        return pen
    def total(theta):
        val = 0.0
        if sn_fun is not None:
            val += sn_fun(theta[:4])
        if bao_fun is not None:
            val += bao_fun(theta)
        return float(val + priors(theta))
    return total

# ---------- minimize ----------
def minimize(chi2, x0, method, max_evals, fix_gamma_zero=False, fit_scales=False):
    import scipy.optimize as opt
    if method=="lbfgsb":
        bounds = [(0.0,3.0), (1.0,6.0), (-math.pi,math.pi), (0.0,1.2)]
        if fix_gamma_zero: bounds[-1] = (0.0,0.0)
        if fit_scales:
            bounds += [(0.6,1.6), (-0.25,0.25), (-0.25,0.25)]  # s_all, dDM, dDH
        return opt.minimize(chi2, x0, method="L-BFGS-B",
                            bounds=bounds, options={"maxfun":max_evals})
    if method=="nelder":
        return opt.minimize(chi2, x0, method="Nelder-Mead", options={"maxfev":max_evals})
    return opt.minimize(chi2, x0, method="Powell", options={"maxfev":max_evals})

# ---------- residuals ----------
def dump_bao_residuals(bao_csv, bao_cov, best):
    import numpy.linalg as la
    df = pd.read_csv(bao_csv)
    y   = df["y_data"].to_numpy(float); z=df["z"].to_numpy(float); kind=df["kind"].astype(str).to_numpy()
    C   = pd.read_csv(bao_cov, header=None).to_numpy(float); L=la.cholesky(C)
    dm  = DM_over_rd(z, best["H0"], best["Om"], best["rd"])
    dh  = DH_over_rd(z, best["H0"], best["Om"], best["rd"])
    mod = M_res(z, best["A"], best["f"], best["phi"], best["gamma"])
    s_all = best.get("s_all",1.0); dDM = best.get("delta_DM",0.0); dDH = best.get("delta_DH",0.0)
    sDM = s_all*(1.0+dDM); sDH = s_all*(1.0+dDH)
    yth = np.where(kind=="DM", sDM*mod*dm, sDH*mod*dh)
    res = y-yth; wres = la.solve(L,res)
    print(f"||wres||^2 (BAO chi2) = {float(wres@wres)}")
    for k in ["DM","DH"]:
        idx=(kind==k); sub=res[idx]
        print(f"{k} N={idx.sum()} mean={sub.mean()} rms={np.sqrt((sub**2).mean())}")

# ---------- main ----------
def main():
    args = parse_args()

    # load
    sn_pack  = load_sn_pack(args.sn_csv, args.sn_cov, args.diag_only)
    bao_pack = load_bao_pack(args.bao_csv, args.bao_cov, args.diag_only)

    # optional SN readability rescale
    theta0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], float)
    if (not args.bao_only) and args.auto_rescale_sn and (not args.diag_only):
        rescale_sn_pack_to_diag(sn_pack, args.H0, args.Om, theta0)

    sn_fun  = None if args.bao_only else make_sn_fun(sn_pack, args.H0, args.Om)
    bao_fun = make_bao_fun_resonant(bao_pack, args.H0, args.Om, args.rd, fit_scales=args.fit_bao_scales)

    theta = [args.A0, args.f0, args.phi0, args.gamma0]
    if args.fit_bao_scales:
        theta += [1.0, 0.0, 0.0]  # s_all, delta_DM, delta_DH
    x0 = np.array(theta, float)

    chi2 = make_objective(
        sn_fun, bao_fun,
        prior_A_sigma=args.prior_A_sigma, prior_gamma_sigma=args.prior_gamma_sigma,
        prior_f_mean=args.prior_f_mean, prior_f_sigma=args.prior_f_sigma,
        fit_scales=args.fit_bao_scales, prior_sall_sigma=args.prior_sall_sigma,
        prior_delta_sigma=args.prior_delta_sigma
    )

    t0=time.time()
    res = minimize(chi2, x0, args.optimizer, args.max_evals,
                   fix_gamma_zero=args.fix_gamma_zero, fit_scales=args.fit_bao_scales)
    t1=time.time()

    A,f,phi,gamma = map(float, res.x[:4])
    best = {"A":A,"f":f,"phi":phi,"gamma":gamma,"H0":args.H0,"Om":args.Om,"rd":args.rd}
    if args.fit_bao_scales:
        best["s_all"] = float(res.x[4])
        best["delta_DM"] = float(res.x[5])
        best["delta_DH"] = float(res.x[6])

    chi2_sn  = float(sn_fun(res.x[:4])) if sn_fun is not None else 0.0
    chi2_bao = float(bao_fun(res.x))

    npar = 4 + (3 if args.fit_bao_scales else 0)
    ndof = int(sn_pack["C"].shape[0]*(0 if args.bao_only else 1) + bao_pack["C"].shape[0] - npar)

    out = {"best_params":best, "chi2":float(res.fun), "ndof":ndof,
           "success":bool(res.success), "message":str(res.message),
           "elapsed_s": t1-t0, "nfev": int(getattr(res,"nfev",0)),
           "breakdown":{"chi2_sn":chi2_sn, "chi2_bao":chi2_bao}}
    p = Path(args.out_prefix).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote {p}")

    if args.dump_residuals:
        dump_bao_residuals(args.bao_csv, args.bao_cov, best)

if __name__=="__main__":
    main()
