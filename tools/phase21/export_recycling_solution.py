#!/usr/bin/env python3
"""
Export Phase-21 recycling solution (JSON or CSV) to a tidy CSV + plot.

- Accepts either JSON (dict-of-arrays or nested) or CSV.
- Recursively finds array fields; chooses an axis (z/a/lookback_time/monotonic).
- Selects density-like series (keys containing 'rho' or 'density').
- Writes a clean CSV and a plot.

Usage:
  python3 tools/phase21/export_recycling_solution.py \
    --in outputs/phase21/recycling_solution.csv \
    --out-csv outputs/phase21/recycling_solution.table.csv \
    --out-plot plots/phase21/recycling_components.png
"""
import argparse, json, csv, os, sys, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out-csv", required=True)
p.add_argument("--out-plot", required=True)
p.add_argument("--axis-pref", nargs="*", default=["z","redshift","a","scale_factor","lookback_time","lookback_gyr","t_lb","time_gyr","time"])
args=p.parse_args()

def read_maybe_json(path):
    txt=open(path,"r",encoding="utf-8").read().strip()
    if not txt:
        sys.exit(f"[error] Empty file: {path}")
    if txt[0] in "{[":
        try:
            return "json", json.loads(txt)
        except Exception:
            pass
    return "csv", None  # we'll load via pandas

def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k,v in obj.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            flatten(kk, v, out)
    else:
        out.append((prefix, obj))

kind, J = read_maybe_json(args.inp)

if kind == "csv":
    # Already a CSV; try to use it directly
    df = pd.read_csv(args.inp)
else:
    # Build dict of arrays from JSON
    items=[]
    flatten("", J, items)
    # collect array-like leaves
    arrays={k:v for k,v in items if isinstance(v, list)}
    # scalars to add to every row later if needed
    scalars={k:v for k,v in items if not isinstance(v, list) and not isinstance(v, dict)}
    if not arrays:
        sys.exit("[error] No array fields found in JSON; cannot build a table.")

    # choose the modal length among arrays
    from collections import Counter
    lens=[len(v) for v in arrays.values()]
    L = Counter(lens).most_common(1)[0][0]
    cols=[k for k,v in arrays.items() if len(v)==L]

    # Build rows
    rows=[]
    for i in range(L):
        r={}
        for k in cols:
            r[k]=arrays[k][i]
        rows.append(r)
    df = pd.DataFrame(rows)

# Choose axis column
def normalize(s): return s.lower().replace("_","").replace("-","").replace(" ","")
cands = args.axis_pref
axis_col=None
norm_cols = {normalize(c): c for c in df.columns}
for name in cands:
    n=normalize(name)
    if n in norm_cols:
        axis_col = norm_cols[n]; break

def is_monotonic_numeric(series):
    s=pd.to_numeric(series, errors="coerce").dropna().values
    if s.size<max(5, len(series)//10): return False
    d=np.diff(s)
    return (d>0).all() or (d<0).all()

if axis_col is None:
    # fallback: find first monotonic numeric column
    for c in df.columns:
        if is_monotonic_numeric(df[c]): axis_col=c; break
if axis_col is None:
    # absolute last resort: first column
    axis_col=df.columns[0]

# If axis is scale factor, add z
zcol = axis_col
if normalize(axis_col) in ("a","scalefactor","ascale"):
    z = 1.0/pd.to_numeric(df[axis_col], errors="coerce") - 1.0
    df.insert(0,"z",z)
    zcol="z"

# Pick density-like series
def looks_density(c):
    n=c.lower()
    return ("rho" in n or "density" in n or n.startswith("rho_") or n.endswith("_rho"))

ycols=[c for c in df.columns if c!=axis_col and looks_density(c)]
if not ycols:
    # take other numeric columns except axis
    num=df.select_dtypes(include=[np.number]).columns.tolist()
    ycols=[c for c in num if c!=axis_col]

if not ycols:
    # if still none, give a friendly error with columns
    print("Columns:", list(df.columns))
    sys.exit("[error] No plottable numeric columns found.")

# Write clean CSV
os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
df.to_csv(args.out_csv, index=False)
print(f"[ok] Wrote {args.out_csv} with columns: {list(df.columns)}; using axis '{zcol if 'z' in df.columns else axis_col}' and {len(ycols)} series.")

# Plot
x = df[zcol] if "z" in df.columns else df[axis_col]
plt.figure()
for c in ycols:
    try:
        plt.plot(pd.to_numeric(x, errors="coerce"),
                 pd.to_numeric(df[c], errors="coerce"), label=c)
    except Exception:
        pass
plt.xlabel("z" if "z" in df.columns else axis_col)
plt.ylabel("density (CSV units)")
plt.legend()
os.makedirs(os.path.dirname(args.out_plot) or ".", exist_ok=True)
plt.savefig(args.out_plot, dpi=160, bbox_inches="tight")
print(f"[ok] Wrote {args.out_plot}")
