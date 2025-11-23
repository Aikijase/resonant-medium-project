#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Per-point chi^2 contributions using full covariance.")
    ap.add_argument("--dump", default="outputs/bao_fit_LCDM.csv", help="Per-entry dump from runner")
    ap.add_argument("--cov",  default="bao_covariance.csv",      help="Covariance CSV")
    ap.add_argument("--out",  default="outputs/bao_contribs.csv",help="Output CSV with contributions")
    args = ap.parse_args()

    df = pd.read_csv(args.dump)
    C  = np.loadtxt(args.cov, delimiter=",").astype(float)

    y  = df["y_data"].to_numpy(float)
    ym = df["y_model"].to_numpy(float)
    r  = y - ym
    n  = len(r)

    if C.shape[0] != n or C.shape[1] != n:
        raise SystemExit(f"Cov size {C.shape} != N {n} (stacking mismatch?)")

    # Stable inverse (tiny jitter)
    eps = 1e-12 * float(np.median(np.diag(C)))
    Cf  = C + eps * np.eye(n)
    try:
        from scipy.linalg import cho_factor, cho_solve
        cf = cho_factor(Cf, lower=True, check_finite=False)
        Ci_r = cho_solve(cf, r, check_finite=False)
    except Exception:
        Ci   = np.linalg.inv(Cf)
        Ci_r = Ci @ r

    # Elementwise contribution under full covariance: c_i = r_i * (C^{-1} r)_i
    contrib = r * Ci_r

    dfc = df.copy()
    dfc["chi2_contrib"] = contrib
    dfc = dfc[["z","kind","y_data","y_model","residual","chi2_contrib"]]

    # Save full table
    dfc_sorted = dfc.sort_values("chi2_contrib", ascending=False).reset_index(drop=True)
    dfc_sorted.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    # Pretty print top contributors and grouped sums
    def f6(x):
        try:
            return f"{float(x):.6f}"
        except:
            return str(x)

    print("\nTop contributors:")
    print(dfc_sorted.head(8).to_string(index=False, formatters={
        "z": f6, "y_data": f6, "y_model": f6, "residual": f6, "chi2_contrib": f6
    }))

    g_kind = dfc.groupby("kind")["chi2_contrib"].sum().sort_values(ascending=False)
    g_z    = dfc.groupby("z")["chi2_contrib"].sum().sort_values(ascending=False)

    print("\nBy kind (sum of contributions):")
    for k, v in g_kind.items():
        print(f"  {k}: {v:.6f}")

    print("\nBy redshift (sum of contributions):")
    for z, v in g_z.items():
        print(f"  z={z:.3f}: {v:.6f}")

if __name__ == "__main__":
    main()
