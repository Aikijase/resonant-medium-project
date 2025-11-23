#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, json

def inv_with_jitter(C):
    eps = 1e-12 * float(np.median(np.diag(C)))
    Cf  = C + eps*np.eye(C.shape[0])
    try:
        from scipy.linalg import cho_factor, cho_solve
        cf = cho_factor(Cf, lower=True, check_finite=False)
        def Ci(v): return cho_solve(cf, v, check_finite=False)
        Ci_mat = None
    except Exception:
        Ci_mat = np.linalg.inv(Cf)
        def Ci(v): return Ci_mat @ v
    return Ci

def main():
    ap = argparse.ArgumentParser(description="Two-β GLS fit: β_DM and β_DH (diagnostic).")
    ap.add_argument("--dump", default="outputs/bao_fit_LCDM.csv")
    ap.add_argument("--cov",  default="bao_covariance.csv")
    ap.add_argument("--meta", default="outputs/bao_fit_LCDM_meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.dump)
    C  = np.loadtxt(args.cov, delimiter=",").astype(float)
    meta = json.load(open(args.meta))
    beta1 = float(meta["beta"])  # single-β from your main run

    y  = df["y_data"].to_numpy(float)
    m0 = df["y_model"].to_numpy(float) / max(beta1,1e-300)  # pre-β model
    kinds = df["kind"].astype(str).to_numpy()

    # Design matrix with two columns
    X = np.zeros((len(y), 2), float)
    X[kinds=="DM",0] = m0[kinds=="DM"]
    X[kinds=="DH",1] = m0[kinds=="DH"]

    Ci = inv_with_jitter(C)
    XtCi = np.vstack([Ci(X[:,0]), Ci(X[:,1])])  # 2 x N
    F = X.T @ XtCi.T  # (X^T C^-1 X)  -> shape (2,2)
    b = np.linalg.solve(F, X.T @ Ci(y))  # β_DM, β_DH

    yhat = X @ b
    r = y - yhat
    chi2 = float(r @ Ci(r))
    dof  = max(len(y) - 2, 1)
    red  = chi2 / dof

    print(f"β_DM={b[0]:.6f}, β_DH={b[1]:.6f},   χ²={chi2:.3f}, dof≈{dof}, χ²/dof={red:.3f}")
    print(f"(single-β from main run was β={beta1:.6f})")

if __name__ == "__main__":
    main()
