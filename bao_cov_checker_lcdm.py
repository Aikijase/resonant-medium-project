import numpy as np
import pandas as pd
from math import sqrt
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve

C_KMS = 299792.458

def Ez_LCDM(z, Om=0.3, Or=0.0):
    Ode = 1.0 - Om - Or
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode)

def DH_over_rd_LCDM(z, H0=70.0, rd=147.1, Om=0.3, Or=0.0):
    H = H0 * Ez_LCDM(z, Om=Om, Or=Or)
    return (C_KMS / H) / rd

def DM_over_rd_LCDM(z, H0=70.0, rd=147.1, Om=0.3, Or=0.0):
    integrand = lambda zz: 1.0 / Ez_LCDM(zz, Om=Om, Or=Or)
    val, _ = quad(integrand, 0.0, z, epsabs=1e-8, epsrel=2e-7, limit=600)
    DC = (C_KMS / H0) * val
    DM = DC
    return DM / rd

def load_meas(path):
    df = pd.read_csv(path)
    if "sigma" not in df.columns:
        df["sigma"] = 1.0
    return df

def build_model(df, H0=70.0, rd=147.1, Om=0.3, Or=0.0):
    y_model = []
    for k, z in zip(df["kind"].values, df["z"].values):
        if k == "DM":
            y_model.append(DM_over_rd_LCDM(z, H0=H0, rd=rd, Om=Om, Or=Or))
        elif k == "DH":
            y_model.append(DH_over_rd_LCDM(z, H0=H0, rd=rd, Om=Om, Or=Or))
        else:
            raise ValueError(f"unknown kind {k}")
    return np.array(y_model, float)

def main(meas_csv, cov_csv, H0=70.0, rd=147.1, Om=0.3, Or=0.0, jitter=0.0):
    df = load_meas(meas_csv)
    y = df["y_data"].values
    C = np.loadtxt(cov_csv, delimiter=",")
    n = C.shape[0]
    print(f"[info] N_meas={len(y)}, N_cov={n}")
    if len(y) != n:
        print("[warn] length mismatch — ordering or selection likely wrong")

    if jitter and jitter > 0:
        C = C + float(jitter)*np.eye(n)

    # model and residual
    y_model = build_model(df, H0=H0, rd=rd, Om=Om, Or=Or)
    r = y - y_model

    try:
        cf = cho_factor(C, lower=True, check_finite=False)
        w = cho_solve(cf, r, check_finite=False)  # C^{-1} r
    except Exception as e:
        print("[error] Cholesky failed:", e); return

    print("\nFirst 10 whitened residuals (C^{-1} r):")
    for i in range(min(10, len(w))):
        print(f"{i:3d}: {w[i]: .3f}   {df['kind'].iloc[i]:2s}  z={df['z'].iloc[i]}  (r={r[i]:.3f})")

    idx = np.argsort(np.abs(w))[-10:][::-1]
    print("\nTop 10 |whitened residuals|:")
    for i in idx:
        print(f"{i:3d}: {w[i]: .3f}   {df['kind'].iloc[i]:2s}  z={df['z'].iloc[i]}  (r={r[i]:.3f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas-csv", required=True)
    ap.add_argument("--cov-csv", required=True)
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--rd", type=float, default=147.1)
    ap.add_argument("--Om", type=float, default=0.3)
    ap.add_argument("--Or", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    args = ap.parse_args()
    main(args.meas_csv, args.cov_csv, H0=args.H0, rd=args.rd, Om=args.Om, Or=args.Or, jitter=args.jitter)
