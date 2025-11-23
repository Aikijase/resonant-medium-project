#!/usr/bin/env python3
"""
Phase 14 — Centerline & curvature between two Si levels.

Takes the contour CSVs produced by phase14_contour_width.py for a pair (low, high),
rebuilds the min/max envelopes per omega2, computes:
  - centerline Kphi_mid(omega2) = 0.5 * [Kphi_max(high) + Kphi_min(low)]
  - width(omega2) = Kphi_max(high) - Kphi_min(low)
Fits a quadratic Kphi_mid = a + b*(w2 - w0) + c*(w2 - w0)^2 and reports curvature c and R^2.

Outputs:
  - centerline CSV: omega2, Kphi_mid, width
  - centerline PNG plot
  - prints fit coefficients & metrics
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def envelope_from_contour(csv_path: str, wbins=900) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna()
    if df.empty:
        return pd.DataFrame(columns=["wbin","omega2","Kphi_min","Kphi_max"])
    wmin, wmax = float(df["omega2"].min()), float(df["omega2"].max())
    if wmin == wmax:
        df["wbin"] = 1
    else:
        bins = np.linspace(wmin, wmax, wbins)
        wbin = np.clip(np.digitize(df["omega2"].to_numpy(), bins), 1, len(bins)-1)
        df = df.assign(wbin=wbin)
    env = (df.groupby("wbin")
             .agg(omega2=("omega2","mean"),
                  Kphi_min=("Kphi","min"),
                  Kphi_max=("Kphi","max"))
             .reset_index()
             .dropna())
    return env

def merge_envelopes(lo_env: pd.DataFrame, hi_env: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(lo_env, hi_env, on="wbin", suffixes=("_lo","_hi"))
    if m.empty: return m
    m["omega2"] = 0.5*(m["omega2_lo"] + m["omega2_hi"])
    m["width"]  = m["Kphi_max_hi"] - m["Kphi_min_lo"]
    m["Kphi_mid"] = 0.5*(m["Kphi_max_hi"] + m["Kphi_min_lo"])
    keep = ["omega2","Kphi_mid","width"]
    m = m.dropna(subset=keep).sort_values("omega2")[keep]
    return m

def fit_quadratic(x, y):
    """Fit y ~ a + b*(x-x0) + c*(x-x0)^2; return (a,b,c,x0,R2)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x0 = 0.5*(x.min()+x.max())
    X = np.column_stack([np.ones_like(x), (x-x0), (x-x0)**2])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    R2 = 1.0 - ss_res/max(ss_tot, 1e-12)
    a,b,c = map(float, coef)
    return a,b,c,x0,R2

def plot_centerline(df, out_png, low, high, a,b,c,x0,R2):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    ax.plot(df["omega2"], df["Kphi_mid"], lw=1.8, label="centerline (midpoint)")
    # overlay fit
    xx = np.linspace(df["omega2"].min(), df["omega2"].max(), 400)
    yy = a + b*(xx-x0) + c*(xx-x0)**2
    ax.plot(xx, yy, ls="--", lw=1.2, label=f"quadratic fit (R²={R2:.3f})")
    ax2 = ax.twinx()
    ax2.plot(df["omega2"], df["width"], lw=1.0, alpha=0.6, label="width", color="tab:gray")
    ax.set_xlabel(r"$\omega^2$")
    ax.set_ylabel(r"$K_\phi$ (centerline)")
    ax2.set_ylabel(r"width in $K_\phi$")
    ax.set_title(f"Centerline & width between Si={low:.2f} and Si={high:.2f}")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contour-low", required=True, help="CSV from phase14_contour_width: *_contour_si<low>.csv")
    ap.add_argument("--contour-high", required=True, help="CSV from phase14_contour_width: *_contour_si<high>.csv")
    ap.add_argument("--low", type=float, required=True)
    ap.add_argument("--high", type=float, required=True)
    ap.add_argument("--wbins", type=int, default=900)
    ap.add_argument("--outdir", default="outputs/phase14")
    ap.add_argument("--prefix", default="p14_edge")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    lo_env = envelope_from_contour(args.contour_low, args.wbins)
    hi_env = envelope_from_contour(args.contour_high, args.wbins)
    merged = merge_envelopes(lo_env, hi_env)
    if merged.empty:
        print("ERROR: no overlapping bins; densify sampling and re-run.", file=sys.stderr)
        sys.exit(2)

    # Save CSV
    out_csv = os.path.join(args.outdir, f"{args.prefix}_centerline_si{args.low:0.3f}_to_si{args.high:0.3f}.csv")
    merged.to_csv(out_csv, index=False)

    # Fit & plot
    a,b,c,x0,R2 = fit_quadratic(merged["omega2"], merged["Kphi_mid"])
    out_png = os.path.join(args.outdir, f"{args.prefix}_centerline_si{args.low:0.3f}_to_si{args.high:0.3f}.png")
    plot_centerline(merged, out_png, args.low, args.high, a,b,c,x0,R2)

    print("centerline_csv:", out_csv)
    print("centerline_png:", out_png)
    print(f"quadratic_fit: a={a:.6f}  b={b:.6f}  c={c:.6f}  (around w0={x0:.6f})  R2={R2:.4f}")
    print(f"curvature_c (quadratic coefficient): {c:.6f}")

if __name__ == "__main__":
    raise SystemExit(main())
