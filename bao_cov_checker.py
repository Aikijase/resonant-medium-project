import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

def load_measurements(path):
    df = pd.read_csv(path)
    if "sigma" not in df.columns:
        df["sigma"] = 1.0
    return df

def main(meas_csv, cov_csv, jitter=0.0):
    df = load_measurements(meas_csv)
    y = df["y_data"].values
    C = np.loadtxt(cov_csv, delimiter=",")
    n_cov = C.shape[0]

    print(f"[info] measurement length = {len(y)}, covariance size = {n_cov}")
    if len(y) != n_cov:
        print("[warn] length mismatch! Need to reorder or adjust vector.")
    
    # Add optional jitter to ensure SPD
    if jitter and jitter > 0:
        C = C + float(jitter) * np.eye(n_cov)

    try:
        cf = cho_factor(C, lower=True, check_finite=False)
        # whitened “residuals”: C^{-1} * y  (we just use y here to inspect scaling)
        w = cho_solve(cf, y, check_finite=False)
    except Exception as e:
        print("[error] Cholesky failed:", e)
        return

    print("\nFirst 10 whitened entries:")
    for i in range(min(10, len(w))):
        print(f"{i:3d}: {w[i]: .5f}   kind={df['kind'].iloc[i]:2s}   z={df['z'].iloc[i]}")

    big = np.argsort(np.abs(w))[-10:][::-1]
    print("\n10 largest |whitened| entries:")
    for i in big:
        print(f"{i:3d}: {w[i]: .3f}   kind={df['kind'].iloc[i]:2s}   z={df['z'].iloc[i]}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas-csv", required=True)
    ap.add_argument("--cov-csv", required=True)
    ap.add_argument("--jitter", type=float, default=0.0)
    args = ap.parse_args()
    main(args.meas_csv, args.cov_csv, jitter=args.jitter)
