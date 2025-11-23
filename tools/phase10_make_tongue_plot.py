#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def guess_cols(df):
    # Try common variants
    col_map = {}
    for want, cands in {
        "omega2": ["omega2","ω2","w2","omega_2","omega","w"],
        "kphi_min": ["Kphi_min_at_thr","Kphi_min_est","Kphi_min","Kφ_min","kphi_min","Kphi","Kφ","kphi"],
        "seed": ["seed","rng_seed","random_seed"],
        "sbar": ["sbar","s̄","sync","sync_index","mean_sync"]
    }.items():
        for c in cands:
            if c in df.columns:
                col_map[want] = c
                break
    if "omega2" not in col_map or "kphi_min" not in col_map:
        raise ValueError(f"Could not find required columns in CSV. Found: {list(df.columns)}")
    return col_map

def quad_fit(x, y, x0=2.8):
    # Fit Kphi_min(ω2) = a + b(ω2-x0) + c(ω2-x0)^2
    X = np.column_stack([np.ones_like(x), (x - x0), (x - x0)**2])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    # Vertex: x_v = x0 - b/(2c)
    xv = x0 - b/(2*c) if c != 0 else np.nan
    yv = a + b*(xv - x0) + c*(xv - x0)**2 if np.isfinite(xv) else np.nan
    return (a, b, c, xv, yv)

def ci95(arr):
    arr = np.asarray(arr)
    m = np.nanmean(arr)
    if len(arr) < 2:
        return m, np.nan, np.nan
    s = np.nanstd(arr, ddof=1)
    lo = m - 1.96*s/np.sqrt(len(arr))
    hi = m + 1.96*s/np.sqrt(len(arr))
    return m, lo, hi

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/phase10/boundary_scan_seeds_adaptive.csv")
    p.add_argument("--out-prefix", default="outputs/phase10/tongue_master")
    p.add_argument("--x0", type=float, default=2.8, help="Expansion point for quadratic fit")
    p.add_argument("--shade-lock", action="store_true", help="Shade lock region above boundary")
    args = p.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    cols = guess_cols(df)
    w = df[cols["omega2"]].values.astype(float)
    k = df[cols["kphi_min"]].values.astype(float)

    # If multiple seeds per ω2, compute mean & CI
    if "seed" in cols:
        g = df.groupby(cols["omega2"])
        agg = g[cols["kphi_min"]].apply(list).reset_index()
        w_unique = agg[cols["omega2"]].values.astype(float)
        means, los, his = [], [], []
        for arr in agg[cols["kphi_min"]].values:
            m, lo, hi = ci95(arr)
            means.append(m); los.append(lo); his.append(hi)
        w_plot = w_unique
        k_plot = np.array(means)
        lo_plot = np.array(los)
        hi_plot = np.array(his)
        has_ci = True
    else:
        # No seeds column—treat as single sample per ω2
        w_plot = w
        k_plot = k
        lo_plot = np.full_like(k_plot, np.nan, dtype=float)
        hi_plot = np.full_like(k_plot, np.nan, dtype=float)
        has_ci = False

    # Quadratic fit
    a, b, c, w_min, k_min = quad_fit(w_plot, k_plot, x0=args.x0)

    # Sort for plotting
    idx = np.argsort(w_plot)
    w_plot = w_plot[idx]; k_plot = k_plot[idx]
    lo_plot = lo_plot[idx]; hi_plot = hi_plot[idx]

    # Smooth fitted curve over a fine grid
    wfine = np.linspace(w_plot.min(), w_plot.max(), 600)
    kfit = a + b*(wfine - args.x0) + c*(wfine - args.x0)**2

    # Plot
    plt.figure(figsize=(7.0, 4.6))
    if has_ci and np.isfinite(lo_plot).any():
        plt.fill_between(w_plot, lo_plot, hi_plot, alpha=0.15, label="95% CI (across seeds)")

    plt.plot(w_plot, k_plot, "o", label="Boundary samples", ms=4)
    plt.plot(wfine, kfit, "-", linewidth=2.0, label="Quadratic fit")

    if args.shade_lock:
        # Shade region above the fitted boundary (locked region)
        ymax = max(np.nanmax(k_plot), np.nanmax(kfit))
        plt.fill_between(wfine, kfit, ymax*1.02, alpha=0.08, label="Lock region (shaded)")

    plt.xlabel(r"$\omega_2$")
    plt.ylabel(r"$K_{\phi,\min}$")
    title = "Phase-10 Arnold Tongue — Boundary & Fit"
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    out_png = Path(f"{args.out_prefix}.png")
    out_pdf = Path(f"{args.out_prefix}.pdf")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    print(f"[phase10] Wrote: {out_png}")
    print(f"[phase10] Wrote: {out_pdf}")
    print(f"[phase10] Fit: Kphi_min(ω2) = {a:.4f} + {b:.4f}(ω2-{args.x0:.3f}) + {c:.4f}(ω2-{args.x0:.3f})^2")
    if np.isfinite(w_min) and np.isfinite(k_min):
        print(f"[phase10] Estimated minimum at ω2 ≈ {w_min:.4f}, Kphi_min ≈ {k_min:.4f}")

if __name__ == "__main__":
    main()
