#!/usr/bin/env python3
"""
Joint BAO+SN fit with optional resonant modulation and flexible BAO scale modes.

Modes:
- Default: fit independent s_DM, s_DH
- --shared-bao-scale: fit one shared scale s_BAO
- --bao-only: ignore SN
"""

from pathlib import Path
import sys, json, math, time, argparse, inspect
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fit_resonant_rd as J

C_LIGHT = 299792.458  # km/s

def _kw_for(fn, ctx):
    sig = inspect.signature(fn)
    return {k: v for k, v in ctx.items() if k in sig.parameters}



# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bao-csv", required=True)
    p.add_argument("--bao-cov", required=True)
    p.add_argument("--sn-csv", required=True)
    p.add_argument("--sn-cov", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--diag-only", action="store_true")

    p.add_argument("--H0", type=float, default=70.0)
    p.add_argument("--Om", type=float, default=0.3)
    p.add_argument("--rd", type=float, default=147.1)

    p.add_argument("--optimizer", default="lbfgsb", choices=["lbfgsb","nelder","powell"])
    p.add_argument("--max-evals", type=int, default=350)

    p.add_argument("--A0", type=float, default=0.4)
    p.add_argument("--f0", type=float, default=2.4)
    p.add_argument("--phi0", type=float, default=1.1)
    p.add_argument("--gamma0", type=float, default=0.0)

    p.add_argument("--prior-A-sigma",     type=float, default=0.6)
    p.add_argument("--prior-gamma-sigma", type=float, default=0.30)
    p.add_argument("--prior-f-mean",      type=float, default=4.0)
    p.add_argument("--prior-f-sigma",     type=float, default=0.6)
    p.add_argument("--prior-sdm-sigma",   type=float, default=0.05)
    p.add_argument("--prior-sdh-sigma",   type=float, default=0.05)

    p.add_argument("--fit-bao-scales", action="store_true", default=True)
    p.add_argument("--shared-bao-scale", action="store_true")
    p.add_argument("--fix-gamma-zero", action="store_true")
    p.add_argument("--bao-only", action="store_true")
    p.add_argument("--sn-only", action="store_true")
    p.add_argument("--auto-rescale-sn", action="store_true")
    p.add_argument("--dump-residuals", action="store_true")
    return p.parse_args()


# -------- cosmology helpers ---------
def Ez(z, Om): return np.sqrt(Om*(1+z)**3 + (1-Om))

def DH_over_rd(z, H0, Om, rd):
    return (C_LIGHT/H0)/rd/Ez(z, Om)

def DM_over_rd(z, H0, Om, rd, nz=800):
    z = np.atleast_1d(z)
    Ode = 1-Om
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        if zi <= 0: out[i]=0; continue
        grid = np.linspace(0, zi, nz)
        E = Ez(grid, Om)
        h = grid[1]-grid[0]
        s = E[0]**-1 + E[-1]**-1 + 4*np.sum(E[1:-1:2]**-1) + 2*np.sum(E[2:-2:2]**-1)
        out[i] = (C_LIGHT/H0)*(h/3*s)/rd
    return out

def M_res(z, A, f, phi, gamma):
    delta = A*np.cos(f*np.log1p(z)+phi)*np.exp(-gamma*z)
    return np.maximum(1.0+delta, 1e-9)


# -------- data loaders ----------
def load_sn_pack(csv, cov, diag_only):
    df = pd.read_csv(csv)
    z, mu = df["z"].to_numpy(float), df["mu"].to_numpy(float)
    C = pd.read_csv(cov, header=None).to_numpy(float)
    if diag_only: C = np.diag(np.diag(C))
    L = np.linalg.cholesky(C)
    return {"z":z,"mu":mu,"y":mu,"C":C,"L":L, "R":L, "L_or_R":L, "F":L, "is_prec": False}

def load_bao_pack(csv, cov, diag_only):
    df = pd.read_csv(csv)
    z = df["z"].to_numpy(float)
    y = df["y_data"].to_numpy(float)
    kind = df["kind"].astype(str).to_numpy()
    C = pd.read_csv(cov, header=None).to_numpy(float)
    if diag_only: C = np.diag(np.diag(C))
    L = np.linalg.cholesky(C)
    return {"z":z,"y":y,"kind":kind,"C":C,"L":L}


# -------- model builders ----------
def make_sn_fun(pack, H0, Om):
    fn = J.make_sn_chi2
    ctx = {**pack, 'sn_pack': pack, 'H0': H0, 'Om': Om, 'Om0': Om}
    return fn(**_kw_for(fn, ctx))

def make_bao_fun(pack, H0, Om, rd, *, fit_scales=True, shared_scale=False):
    z, y, kind, L = pack["z"], pack["y"], pack["kind"], pack["L"]
    dm = DM_over_rd(z, H0, Om, rd)
    dh = DH_over_rd(z, H0, Om, rd)
    mDM = kind=="DM"; mDH = kind=="DH"

    def predict(theta):
        A,f,phi,gamma = theta[:4]
        s_DM = s_DH = 1.0
        if fit_scales:
            if shared_scale: s_DM = s_DH = theta[4]
            else: s_DM, s_DH = theta[4], theta[5]
        # zero-mean the cosine so the modulation cannot mimic a constant offset
        # compute raw cosine at the current (A,f,phi,gamma), then subtract its BAO-sample mean
        raw = np.cos(f*np.log1p(z) + phi) * np.exp(-gamma*z)
        raw = raw - raw.mean()
        mod = 1.0 + A * raw
        mod = np.maximum(mod, 1e-9)
        y_th = np.empty_like(y)
        y_th[mDM] = s_DM*mod[mDM]*dm[mDM]
        y_th[mDH] = s_DH*mod[mDH]*dh[mDH]
        return y_th

    def chi2(theta):
        res = y - predict(theta)
        x = np.linalg.solve(L,res)
        return float(x@x)
    return chi2


# -------- objective ----------
def make_objective(sn_fun, bao_fun, *, fit_scales=True, shared_scale=False,
                   prior_A_sigma=0.6, prior_gamma_sigma=0.3,
                   prior_f_mean=4.0, prior_f_sigma=0.6,
                   prior_sdm_sigma=0.05, prior_sdh_sigma=0.05):
    def priors(theta):
        A,f,phi,gamma = theta[:4]
        pen = (A/prior_A_sigma)**2 + (gamma/prior_gamma_sigma)**2
        pen += ((f-prior_f_mean)/prior_f_sigma)**2
        if fit_scales:
            if shared_scale:
                pen += ((theta[4]-1)/prior_sdm_sigma)**2
            else:
                pen += ((theta[4]-1)/prior_sdm_sigma)**2
                pen += ((theta[5]-1)/prior_sdh_sigma)**2
        return pen

    def chi2(theta):
        base = 0
        if sn_fun: base += sn_fun(theta[:4])
        if bao_fun: base += bao_fun(theta)
        return base + priors(theta)
    return chi2


# -------- minimize ----------
def minimize(chi2, x0, method, maxevals, fit_scales, shared_scale, fix_gamma):
    import scipy.optimize as opt, math

    n = len(x0)
    # Base order assumed: [A, f, phi, gamma, (s_shared or s_DM), (s_DH)]
    bounds = [
        (0.0, 2.0),             # A
        (1.0, 6.0),             # f
        (-math.pi, math.pi),    # phi
        (0.0, 3.0),             # gamma  (was 1.0; widened)
    ]

    # If x0 includes scale(s), add their bounds regardless of flags,
    # so bounds always match x0 length.
    if n >= 5:
        bounds.append((0.8, 1.2))   # s_shared or s_DM
    if n >= 6:
        bounds.append((0.8, 1.2))   # s_DH

    if fix_gamma:
        bounds[3] = (0.0, 0.0)

    # Final safety: trim in case n<4 (unlikely) or overbuilt
    bounds = bounds[:n]

    return opt.minimize(
        chi2, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxfun": maxevals}
    )


# -------- main ----------
def main():
    a = parse_args()
    sn_pack = load_sn_pack(a.sn_csv, a.sn_cov, a.diag_only)
    bao_pack = load_bao_pack(a.bao_csv, a.bao_cov, a.diag_only)
    sn_fun  = None if a.bao_only else make_sn_fun(sn_pack, a.H0, a.Om)
    bao_fun = make_bao_fun(bao_pack, a.H0, a.Om, a.rd,
                           fit_scales=a.fit_bao_scales,
                           shared_scale=a.shared_bao_scale)
    if a.sn_only:
        bao_fun = None

    theta = [a.A0,a.f0,a.phi0,a.gamma0]
    if a.fit_bao_scales:
        theta += [1.0] if a.shared_bao_scale else [1.0,1.0]
    x0 = np.array(theta,float)

    chi2 = make_objective(sn_fun, bao_fun,
                          fit_scales=a.fit_bao_scales,
                          shared_scale=a.shared_bao_scale,
                          prior_A_sigma=a.prior_A_sigma,
                          prior_gamma_sigma=a.prior_gamma_sigma,
                          prior_f_mean=a.prior_f_mean,
                          prior_f_sigma=a.prior_f_sigma,
                          prior_sdm_sigma=a.prior_sdm_sigma,
                          prior_sdh_sigma=a.prior_sdh_sigma)

    t0=time.time()
    res=minimize(chi2,x0,a.optimizer,a.max_evals,
                 a.fit_bao_scales,a.shared_bao_scale,a.fix_gamma_zero)
    t1=time.time()

    A,f,phi,gamma=res.x[:4]
    best={"A":float(A),"f":float(f),"phi":float(phi),"gamma":float(gamma),
          "H0":a.H0,"Om":a.Om,"rd":a.rd}
    if a.fit_bao_scales:
        if a.shared_bao_scale:
            best["s_DM"]=best["s_DH"]=float(res.x[4])
        else:
            best["s_DM"]=float(res.x[4])
            best["s_DH"]=float(res.x[5])

    chi2_sn=float(sn_fun(res.x[:4])) if sn_fun else 0.0
    chi2_bao=float(bao_fun(res.x)) if bao_fun else 0.0
    ndof=(0 if a.sn_only else bao_pack['C'].shape[0]) + (0 if a.bao_only else sn_pack['C'].shape[0]) - len(res.x)
    out={"best_params":best,"chi2":float(res.fun),"ndof":int(ndof),
         "breakdown":{"chi2_sn":chi2_sn,"chi2_bao":chi2_bao},
         "elapsed_s":t1-t0,"success":bool(res.success),"message":res.message}

    outpath=Path(a.out_prefix).with_suffix(".json")
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(out,indent=2))
    print(f"Wrote {outpath}")

    if a.dump_residuals:
        print("A,f,phi,gamma =",A,f,phi,gamma)
        if bao_fun:
            print("||wres||^2 (BAO chi2) =", chi2_bao)
            if "s_DM" in best: print("s_DM,s_DH =",best.get("s_DM"),best.get("s_DH"))
        else:
            print("(SN-only run: no BAO residuals to dump)")


if __name__=="__main__":
    main()
