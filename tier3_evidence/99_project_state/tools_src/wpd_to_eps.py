#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, os
ap=argparse.ArgumentParser(description="Convert WebPlotDigitizer CSV (X,Y) → omega,omega0,epsilon,label")
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--omega0", type=float, required=True)
ap.add_argument("--label", default="BEC ridge")
ap.add_argument("--flip-y", action="store_true", help="Use if your Y increases downward in the figure")
a=ap.parse_args()
df = pd.read_csv(a.inp)
# take first two columns as X,Y
cols=list(df.columns)[:2]
x = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy()
y = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy()
n = min(len(x), len(y))
x, y = x[:n], y[:n]
if a.flip_y: y = 1.0 - y
y = np.clip(y, 0.0, None)
m = float(np.nanmax(y)) if n else 1.0
eps = y / (m if m>0 else 1.0)
out = pd.DataFrame({"omega": x, "omega0": a.omega0, "epsilon": eps, "label": a.label})
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
out.to_csv(a.out, index=False)
print(f"Wrote {len(out)} rows → {a.out}")
