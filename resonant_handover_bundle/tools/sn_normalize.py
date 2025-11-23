#!/usr/bin/env python3
import argparse, pathlib as p, pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Normalize Pantheon+ catalog into a SN vector matching the STAT+SYS covariance.")
    ap.add_argument("--in", dest="inp", required=True, help="Pantheon+ catalog CSV (e.g., Pantheon+SH0ES.csv)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV for SN vector (z,mu,sigma)")
    ap.add_argument("--use-mb-corr", action="store_true", help="Use standardized magnitudes m_b_corr (RECOMMENDED for full covariance)")
    ap.add_argument("--use-mu", action="store_true", help="Use MU_SH0ES (for diagonal-only experiments)")
    ap.add_argument("--zcol", default="zHD", help="Redshift column name (default: zHD)")
    args = ap.parse_args()

    inp = p.Path(args.inp).expanduser()
    outp= p.Path(args.out).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if args.use_mb_corr and args.use_mu:
        raise SystemExit("Choose either --use-mb-corr OR --use-mu, not both.")
    use_mb = args.use_mb_corr or not args.use_mu

    z = df[args.zcol if args.zcol in df.columns else "zHD"]

    if use_mb:
        if "m_b_corr" not in df.columns:
            raise SystemExit("m_b_corr not found in catalog.")
        mu = df["m_b_corr"]
        sigma = df.get("m_b_corr_err_DIAG", pd.Series([1.0]*len(df)))
    else:
        if "MU_SH0ES" not in df.columns:
            raise SystemExit("MU_SH0ES not found in catalog.")
        mu = df["MU_SH0ES"]
        sigma = df.get("MU_SH0ES_ERR_DIAG", pd.Series([1.0]*len(df)))

    out = pd.DataFrame({"z": z.values, "mu": mu.values, "sigma": sigma.values})
    out.to_csv(outp, index=False)
    print(f"[ok] wrote {outp} rows: {len(out)}")
    print(out.head(3))

if __name__ == "__main__":
    main()
