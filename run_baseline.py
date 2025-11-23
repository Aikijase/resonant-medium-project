#!/usr/bin/env python3
import os, json, numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from code.models.cosmology_basics import LCDM
from code.loaders.pantheon_plus import try_all_and_match

MANIFEST = "data_manifest.json"

def prepare_sn_problem(z, mu, C):
    # Add a tiny jitter in case C is near-singular
    eps = 1e-9 * np.median(np.diag(C))
    Cj = C + np.eye(C.shape[0]) * eps
    cf = cho_factor(Cj, lower=True, check_finite=False)
    ones = np.ones_like(mu)

    def Ci(v):  # apply C^{-1} via Cholesky solve
        return cho_solve(cf, v, check_finite=False)

    W1 = Ci(ones)
    denom = float(ones @ W1)  # 1^T C^{-1} 1

    def chi2_and_Mopt(Om):
        cosmo = LCDM(H0=70.0, Omega_m=Om)
        mu_th0 = cosmo.distance_modulus(z)  # no M yet
        r0 = mu - mu_th0
        num = float(ones @ Ci(r0))          # 1^T C^{-1} r0
        Mopt = num / denom                  # analytic best-fit M
        r = r0 - Mopt * ones
        chi2 = float(r @ Ci(r))
        return chi2, Mopt

    return chi2_and_Mopt

def run_sn_only(manifest):
    pant_files = [p for p in manifest.get('pantheon_plus', []) if os.path.exists(p)]
    dats = [p for p in pant_files if p.lower().endswith('.dat')]
    covs = [p for p in pant_files if p.lower().endswith('.cov')]
    if not dats or not covs:
        raise FileNotFoundError("Pantheon+ .dat or .cov not found via manifest")
    cov = covs[0]
    z, mu, C, used_dat = try_all_and_match(dats, cov, debug=False)
    print(f"[info] Using DAT: {os.path.basename(used_dat)}")
    print(f"[info] Using COV: {os.path.basename(cov)} (N={len(mu)})")

    chi2_M = prepare_sn_problem(z, mu, C)

    # 1D bounded search over Ωm
    res = minimize_scalar(lambda Om: chi2_M(Om)[0], bounds=(0.05, 0.6), method="bounded")
    Om_best = float(res.x)
    chi2_best, M_best = chi2_M(Om_best)
    dof = len(z) - 1  # parameters: (Ωm + M) but M was profiled (analytically), still counts as 1 dof

    return {
        "Omega_m": Om_best,
        "M": float(M_best),
        "chi2": float(chi2_best),
        "dof": int(dof),
        "chi2_red": float(chi2_best/dof),
        "success": bool(res.success),
        "n_eval": int(res.nfev) if hasattr(res, "nfev") else None,
    }

if __name__ == "__main__":
    if not os.path.exists(MANIFEST):
        raise SystemExit("data_manifest.json not found. Run scripts/check_data.py first.")
    with open(MANIFEST, "r") as f:
        manifest = json.load(f)
    try:
        out = run_sn_only(manifest)
        print("=== Pantheon+ (SN-only) baseline ===")
        for k, v in out.items():
            print(f"{k:10s}: {v}")
    except Exception as e:
        print(f"[WARN] SN-only fit failed: {e}")
