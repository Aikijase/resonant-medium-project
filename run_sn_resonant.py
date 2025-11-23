#!/usr/bin/env python3
import os, json, numpy as np
from math import pi
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar, minimize

# ---------------- utilities ----------------
def prep_problem(z, mu, C):
    eps = 1e-9 * np.median(np.diag(C))
    Cj  = C + np.eye(C.shape[0]) * eps
    cf  = cho_factor(Cj, lower=True, check_finite=False)
    ones = np.ones_like(mu)
    def Ci(v): return cho_solve(cf, v, check_finite=False)
    denom = float(ones @ Ci(ones))
    def profile_M(resid0):
        num  = float(ones @ Ci(resid0))
        Mopt = num/denom
        return Mopt, resid0 - Mopt*ones
    return Ci, profile_M

def chi2_from_mu(Ci, r): return float(r @ Ci(r))

# fixed z-grid (uniform) for fast cumulative integration
def make_grid(z, zpad=0.02, N=2000):
    zmax = float(np.max(z))*(1.0+zpad)
    zg   = np.linspace(0.0, zmax, N)
    dz   = zg[1]-zg[0]
    return zg, dz

# comoving distances via one cumulative trapezoid per parameter eval
C_KMS = 299792.458
H0    = 70.0

def mu_LCDM(z, Om, zg, dz):
    E  = np.sqrt(Om*(1+zg)**3 + (1-Om))
    invE = 1.0/np.maximum(E, 1e-12)
    cum = np.empty_like(zg)
    cum[0] = 0.0
    cum[1:] = np.cumsum(0.5*(invE[1:]+invE[:-1])*dz)
    chi = np.interp(z, zg, cum) * (C_KMS/H0)
    dl  = (1+z)*chi
    return 5.0*np.log10(dl*1e6) - 5.0

def mu_RES(z, Om, A, f, phi, zg, dz):
    a   = 1.0/(1.0+zg)
    ff  = f if abs(f) > 1e-6 else 1e-6
    osc = np.exp(3.0*A/ff*(np.cos(ff*np.log(a)+phi) - np.cos(phi)))
    E2  = Om*(1+zg)**3 + (1-Om)*osc
    E   = np.sqrt(np.maximum(E2, 1e-12))
    invE = 1.0/E
    cum = np.empty_like(zg)
    cum[0] = 0.0
    cum[1:] = np.cumsum(0.5*(invE[1:]+invE[:-1])*dz)
    chi = np.interp(z, zg, cum) * (C_KMS/H0)
    dl  = (1+z)*chi
    return 5.0*np.log10(dl*1e6) - 5.0

# ---------------- main fits ----------------
def fit_LCDM(z, mu, C, zg, dz):
    Ci, prof = prep_problem(z, mu, C)
    def chi2_of_Om(Om):
        r0 = mu - mu_LCDM(z, Om, zg, dz)
        Mopt, r = prof(r0)
        return chi2_from_mu(Ci, r), Mopt
    res = minimize_scalar(lambda Om: chi2_of_Om(Om)[0], bounds=(0.05,0.6), method="bounded")
    Om = float(res.x)
    chi2, Mopt = chi2_of_Om(Om)
    N = len(z); k = 2  # (Ωm, M)
    return {"Om": Om, "M": Mopt, "chi2": chi2, "dof": N-k, "k": k}

def fit_RES(z, mu, C, zg, dz, verbose=True):
    Ci, prof = prep_problem(z, mu, C)
    evals = {"n":0, "best":np.inf}
    def chi2_params(x):
        Om, A, f, phi = x
        if not (0.05<=Om<=0.6 and -0.2<=A<=0.2 and 0.2<=f<=5.0 and 0.0<=phi<=2*pi):
            return 1e12
        r0 = mu - mu_RES(z, Om, A, f, phi, zg, dz)
        Mopt, r = prof(r0)
        val = chi2_from_mu(Ci, r)
        evals["n"] += 1
        if val < evals["best"]:
            evals["best"] = val
            if verbose:
                print(f"[res] eval {evals['n']:4d}  chi2={val:9.3f}  Om={Om:.3f} A={A:.3f} f={f:.3f} phi={phi:.3f}")
        elif verbose and evals["n"] % 20 == 0:
            print(f"[res] eval {evals['n']:4d}  chi2~{val:9.3f}")
        return val

    starts = [
        [0.30,  0.00, 1.0, 0.0],
        [0.28,  0.05, 0.8, 1.0],
        [0.32, -0.05, 1.5, 2.0],
        [0.26,  0.10, 0.4, 0.5],
        [0.34, -0.10, 2.5, 3.0],
    ]
    bnds = [(0.05,0.6), (-0.2,0.2), (0.2,5.0), (0, 2*pi)]
    best_fun, best_x = np.inf, None
    for x0 in starts:
        res = minimize(chi2_params, x0=np.array(x0), bounds=bnds, method="L-BFGS-B", options={"maxfun": 400})
        if res.fun < best_fun:
            best_fun, best_x = res.fun, res.x
    Om, A, f, phi = map(float, best_x)
    # recover M at best
    r0 = mu - mu_RES(z, Om, A, f, phi, zg, dz)
    Mopt, r = prof(r0)
    N = len(z); k = 5  # (Ωm, A, f, phi, M)
    return {"Om": Om, "A": A, "f": f, "phi": phi, "M": float(Mopt), "chi2": float(best_fun), "dof": N-k, "k": k, "n_eval": evals["n"]}

def BIC(chi2, k, N): return chi2 + k*np.log(N)
def AIC(chi2, k):    return chi2 + 2*k

if __name__ == "__main__":
    MANIFEST = "data_manifest.json"
    if not os.path.exists(MANIFEST):
        raise SystemExit("data_manifest.json not found. Run scripts/check_data.py first.")
    with open(MANIFEST, "r") as f:
        man = json.load(f)

    pant = [p for p in man.get("pantheon_plus", []) if os.path.exists(p)]
    dats = [p for p in pant if p.lower().endswith(".dat")]
    covs = [p for p in pant if p.lower().endswith(".cov")]
    if not dats or not covs:
        raise SystemExit("Pantheon+ files not found via manifest.")
    # choose first matching .dat (we already verified earlier)
    dat = sorted(dats)[0]; cov = sorted(covs)[0]
    # lightweight parser tailored to Pantheon+ columns
    import pandas as pd
    df  = pd.read_csv(dat, sep=r"\s+", engine="python", comment="#")
    z   = df["zCMB"].to_numpy(dtype=float)
    mu  = df["MU_SH0ES"].to_numpy(dtype=float)
    C   = np.loadtxt(cov)
    # reshape covariance if needed
    N = mu.size
    if C.ndim == 1 and C.size == N*N + 1 and int(round(C[0])) == N: C = C[1:].reshape(N,N)
    elif C.ndim == 1 and C.size == N*N: C = C.reshape(N,N)
    # grid
    zg, dz = make_grid(z, zpad=0.02, N=2000)

    # baseline ΛCDM
    base = fit_LCDM(z, mu, C, zg, dz)
    # resonant
    resn = fit_RES(z, mu, C, zg, dz, verbose=True)

    out = {
        "LCDM": {**base, "BIC": BIC(base["chi2"], base["k"], N), "AIC": AIC(base["chi2"], base["k"])},
        "RESN": {**resn, "BIC": BIC(resn["chi2"], resn["k"], N), "AIC": AIC(resn["chi2"], resn["k"])},
    }
    print("\n=== SN-only comparison ===")
    print("LCDM :", out["LCDM"])
    print("RESN :", out["RESN"])
    print("\nΔχ² (LCDM−RESN) :", out["LCDM"]["chi2"] - out["RESN"]["chi2"])
    print("ΔBIC            :", out["LCDM"]["BIC"] - out["RESN"]["BIC"])
    print("ΔAIC            :", out["LCDM"]["AIC"] - out["RESN"]["AIC"])
