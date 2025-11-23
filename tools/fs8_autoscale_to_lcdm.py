#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in-csv",  default="outputs/phase2/fs8_eval.csv")
ap.add_argument("--z-col",   default="z")
ap.add_argument("--lcdm-col",default="fs8_fit")
ap.add_argument("--model-col",default="fs8_pred")
ap.add_argument("--z-min", type=float, default=None)
ap.add_argument("--z-max", type=float, default=1.0, help="fit window (trusted)")
ap.add_argument("--out-csv", default="outputs/phase2/fs8_eval_scaled.csv")
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
z = pd.to_numeric(df[args.z_col], errors="coerce").to_numpy()
L = pd.to_numeric(df[args.lcdm_col], errors="coerce").to_numpy()
M = pd.to_numeric(df[args.model_col], errors="coerce").to_numpy()

mask = np.isfinite(z) & np.isfinite(L) & np.isfinite(M) & (L>0) & (M>0)
if args.z_min is not None: mask &= (z >= args.z_min)
if args.z_max is not None: mask &= (z <= args.z_max)

Lm = L[mask]; Mm = M[mask]
if Mm.size == 0 or np.allclose((Mm*Mm).sum(), 0):
    raise SystemExit("No valid points to fit scale — check your inputs.")

s = float((Lm*Mm).sum() / (Mm*Mm).sum())
df["fs8_pred_scaled"] = df[args.model_col] * s
Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out_csv, index=False)

print(f"Fitted scale s = {s:.6e}  using N={Mm.size} points (z<= {args.z_max})")
print(f"Wrote {args.out_csv} with column fs8_pred_scaled")
