#!/usr/bin/env python3
import argparse, sys, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--out", default="plots/phase21/recycling_components.png")
p.add_argument("--axis-col", help="Axis column name (case-insensitive, e.g. z, a, redshift, scale_factor)")
p.add_argument("--axis-col-index", type=int, help="Axis column index (0-based) if you prefer to pick by position")
args=p.parse_args()

df=pd.read_csv(args.csv)

# Print columns for visibility
print("COLUMNS:", list(df.columns))

cols_lower=[c.lower() for c in df.columns]

def case_insensitive_find(name):
    if name is None: return None
    name=name.lower()
    if name in cols_lower:
        return df.columns[cols_lower.index(name)]
    # allow underscores/dashes/spaces differences
    norm=lambda s: s.lower().replace("_","").replace("-","").replace(" ","")
    nname=norm(name)
    for c in df.columns:
        if norm(c)==nname:
            return c
    return None

axis_col=None

# 1) Respect explicit index if provided
if args.axis_col_index is not None:
    try:
        axis_col=df.columns[args.axis_col_index]
        print(f"[info] Using axis by index {args.axis_col_index}: '{axis_col}'")
    except Exception as e:
        sys.exit(f"[error] axis-col-index {args.axis_col_index} is out of range for columns {list(df.columns)}")

# 2) Or respect explicit name (case-insensitive, tolerant)
if axis_col is None and args.axis_col:
    axis_col = case_insensitive_find(args.axis_col)
    if axis_col:
        print(f"[info] Using axis by name: '{axis_col}'")
    else:
        print(f"[warn] --axis-col '{args.axis_col}' not found. Will try auto-detect.")

# 3) Auto-detect by common names
if axis_col is None:
    candidates = ["z","redshift","z_mid","zbin","z_bin","zcenter","z_center",
                  "a","scale_factor","a_scale",
                  "lookback_time","lookback_gyr","t_lb","t_gyr","time_gyr","time",
                  "tau","eta","chi","comoving_chi"]
    for c in candidates:
        if c in cols_lower:
            axis_col = df.columns[cols_lower.index(c)]
            print(f"[info] Auto-detected axis: '{axis_col}'")
            break

# 4) Fallback: first monotonic numeric col that isn’t all-NaN
def is_monotonic_numeric(series):
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if s.size < max(5, len(series)//5):  # at least some finite points
        return False
    d = np.diff(s)
    return np.all(d>0) or np.all(d<0)

if axis_col is None:
    for c in df.select_dtypes(include=[np.number]).columns:
        if is_monotonic_numeric(df[c]):
            axis_col = c
            print(f"[info] Fallback axis: '{axis_col}' (monotonic numeric)")
            break

if axis_col is None:
    # Last resort: first column
    axis_col = df.columns[0]
    print(f"[info] Last-resort axis: '{axis_col}' (first column)")

# Convert a→z if needed
zcol = axis_col
if axis_col.lower() in ("a","scale_factor","a_scale"):
    z = 1.0/pd.to_numeric(df[axis_col], errors="coerce") - 1.0
    df.insert(0, "z", z)
    zcol = "z"
    print("[info] Converted scale factor 'a' to redshift 'z'")

def looks_density(name):
    n=name.lower()
    return ("rho" in n or "density" in n or n.startswith("rho_") or n.endswith("_rho"))

# Choose Y series
ycols = [c for c in df.columns if c!=axis_col and looks_density(c)]
if not ycols:
    ycols = [c for c in df.select_dtypes(include=[np.number]).columns if c!=axis_col]
if not ycols:
    sys.exit("[error] No plottable numeric columns found.")

# Plot
x = df[zcol] if zcol in df.columns else df[axis_col]
import matplotlib
matplotlib.rcParams.update({'figure.figsize': (7.0, 4.2)})
plt.figure()
for c in ycols:
    try:
        plt.plot(pd.to_numeric(x, errors="coerce"),
                 pd.to_numeric(df[c], errors="coerce"), label=c)
    except Exception:
        pass
plt.xlabel("z" if zcol in df.columns else axis_col)
plt.ylabel("density (units as in CSV)")
plt.legend()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
plt.savefig(args.out, dpi=160, bbox_inches="tight")
print(f"[ok] Wrote {args.out} using axis='{zcol if zcol in df.columns else axis_col}' and {len(ycols)} series.")
