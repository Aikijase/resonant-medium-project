#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np

def pick(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low: return low[n]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--frac", type=float, default=0.05, help="fractional error for fs8 (default 5%)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing fs8_err if present")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    zc   = pick(df, ["z","z_eff","redshift"])
    fs8c = pick(df, ["fs8","fsigma8","fσ8","fs8_obs"])
    if not zc or not fs8c:
        raise SystemExit(f"Need z and fs8-like columns. Got {list(df.columns)}")

    # Only fill if missing or overwrite requested
    errc = pick(df, ["fs8_err","sigma_fs8","err","uncertainty"])
    if errc and not args.overwrite:
        print(f"Found existing error column '{errc}'. Use --overwrite to replace.")
        df.rename(columns={zc:"z", fs8c:"fs8", errc:"fs8_err"}).to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
        return

    fs8 = pd.to_numeric(df[fs8c], errors="coerce")
    fs8_err = args.frac * fs8
    out = df.copy()
    out.rename(columns={zc:"z", fs8c:"fs8"}, inplace=True)
    out["fs8_err"] = fs8_err
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with fs8_err = {args.frac:.3f} * fs8")

if __name__ == "__main__":
    main()
