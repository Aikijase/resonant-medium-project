#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="outputs/bao_fit_LCDM.csv", help="Per-entry dump CSV")
    ap.add_argument("--cov",  default="bao_covariance.csv", help="Covariance CSV")
    ap.add_argument("--out",  default="figs/bao_residuals.png", help="Output plot")
    args = ap.parse_args()

    df = pd.read_csv(args.dump)
    C  = np.loadtxt(args.cov, delimiter=",").astype(float)

    y = df["y_data"].to_numpy(float)
    ym = df["y_model"].to_numpy(float)
    r = y - ym
    if C.shape[0] != len(r):
        raise SystemExit(f"Cov size {C.shape} != N {len(r)} (did the stacking match?)")

    # χ² = r^T C^-1 r
    Ci = np.linalg.inv(C)
    chi2 = float(r @ (Ci @ r))
    N = len(r)
    k = 1  # profiled beta ~ 1 param
    dof = max(N - k, 1)
    red = chi2 / dof

    print(f"N = {N}, dof ≈ {dof}")
    print(f"χ² = {chi2:.3f},  χ²/dof = {red:.3f}")

    # Residuals vs z (units of the observable)
    plt.figure()
    plt.scatter(df["z"], r, label="residuals")
    plt.axhline(0, ls="--", lw=1)
    plt.xlabel("z"); plt.ylabel("y_data - y_model")
    plt.title("BAO residuals")
    plt.legend()
    plt.tight_layout(); plt.savefig(args.out, dpi=140); plt.close()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
