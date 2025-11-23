import argparse, pandas as pd, sys, pathlib as p
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
ap.add_argument("--use-mb-corr", action="store_true")
ap.add_argument("--zcol", default="zHD")
args = ap.parse_args()
inp = p.Path(args.inp).expanduser()
df  = pd.read_csv(inp)
cols = {c.lower(): c for c in df.columns}
def pick(name, *alts):
    for n in (name,)+alts:
        if n in df: return n
        if n.lower() in cols: return cols[n.lower()]
    raise SystemExit(f"[error] missing column like {name} in {list(df.columns)[:10]}...")
zcol = pick(args.zcol)
if args.use_mb_corr:
    mu   = pick("m_b_corr")
    sig  = pick("m_b_corr_err_DIAG")
else:
    mu   = pick("MU_SH0ES","mu","distance_modulus")
    sig  = pick("MU_SH0ES_ERR_DIAG","muerr","sigma")
out = pd.DataFrame({"z": df[zcol].values,
                    "mu": df[mu].values,
                    "sigma": df[sig].values})
out.to_csv(p.Path(args.out).expanduser(), index=False)
print("[ok] wrote", args.out, "rows:", len(out))
