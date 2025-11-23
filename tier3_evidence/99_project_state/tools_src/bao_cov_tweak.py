#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, os

def main():
    ap = argparse.ArgumentParser(description="Add fractional systematic in quadrature to selected rows.")
    ap.add_argument("--dump", required=True, help="Per-entry CSV (e.g. outputs/bao_fit_LCDM.csv)")
    ap.add_argument("--cov-in", required=True, help="Original covariance CSV")
    ap.add_argument("--cov-out", required=True, help="Output covariance CSV")
    ap.add_argument("--z", type=float, required=True, help="Redshift to target (exact or within --tol)")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tolerance for matching z")
    ap.add_argument("--frac", type=float, default=0.07, help="Add (frac*y_model)^2 to diagonal")
    ap.add_argument("--kinds", type=str, default="", help="Comma list (e.g. 'DH' or 'DM,DH'); empty = all kinds")
    args = ap.parse_args()

    df = pd.read_csv(args.dump)
    C  = np.loadtxt(args.cov_in, delimiter=",").astype(float)
    if C.shape[0] != len(df): raise SystemExit(f"[cov] shape {C.shape} != N {len(df)}")

    z     = df["z"].to_numpy(float)
    kind  = df["kind"].astype(str).str.upper().to_numpy()
    ym    = df["y_model"].to_numpy(float)

    sel = np.abs(z - args.z) <= args.tol
    if args.kinds.strip():
        allowed = {k.strip().upper() for k in args.kinds.split(",")}
        sel &= np.isin(kind, list(allowed))

    idx = np.where(sel)[0]
    if len(idx) == 0:
        raise SystemExit(f"No rows with z≈{args.z} and kinds={args.kinds or 'ALL'}")

    C2 = C.copy()
    for i in idx:
        C2[i,i] += (args.frac * ym[i])**2

    outdir = os.path.dirname(args.cov_out)
    if outdir: os.makedirs(outdir, exist_ok=True)
    np.savetxt(args.cov_out, C2, delimiter=",")
    print(f"[tweak] matched rows: {idx.tolist()}, kinds={args.kinds or 'ALL'}, frac={args.frac}")
    print(f"[tweak] wrote {args.cov_out}")

if __name__ == "__main__":
    main()
