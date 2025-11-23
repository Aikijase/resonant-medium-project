#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def guess_cols(df):
    col_map = {}
    for want, cands in {
        "omega2": ["omega2","ω2","w2","omega_2","omega","w"],
        "kphi_min": ["Kphi_min_at_thr","Kphi_min_est","Kphi_min","Kφ_min","kphi_min","Kphi","Kφ","kphi"],
        "seed": ["seed","rng_seed","random_seed"],
        "sbar": ["sbar","s̄","sync","sync_index","mean_sync"],
    }.items():
        for c in cands:
            if c in df.columns:
                col_map[want] = c if want not in col_map else col_map[want]
    if "omega2" not in col_map or "kphi_min" not in col_map:
        raise ValueError(f"Could not find required columns. Found: {list(df.columns)}")
    return col_map

def quad_fit(x, y, x0=2.8):
    X = np.column_stack([np.ones_like(x), (x-x0), (x-x0)**2])
    a,b,c = np.linalg.lstsq(X, y, rcond=None)[0]
    xv = x0 - b/(2*c) if c != 0 else np.nan
    yv = a + b*(xv-x0) + c*(xv-x0)**2 if np.isfinite(xv) else np.nan
    return a,b,c,xv,yv

def ci95(arr):
    arr = np.asarray(arr, float)
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
    p.add_argument("--out-prefix", default="outputs/phase10/tongue_master_plus")
    p.add_argument("--x0", type=float, default=2.8)
    p.add_argument("--shade-lock", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    cols = guess_cols(df)

    # --- Robust filtering for wide scans ---
    # Coerce boundary column to numeric and drop invalid/unreached rows
    df[cols["kphi_min"]] = pd.to_numeric(df[cols["kphi_min"]], errors="coerce")
    status_col = None
    for c in ["status","Status","STATE"]:
        if c in df.columns:
            status_col = c
            break
    mask = df[cols["kphi_min"]].apply(np.isfinite)
    if status_col is not None:
        mask &= df[status_col].isin(["ok","at_lower"])
    df = df[mask].copy()
    if df.empty:
        print("[phase10+] No valid boundary rows after filtering; check your CSV (status and Kphi_min).")
        # Still write empty-fit artifacts to avoid breaking make
        Path(args.out_prefix + ".fit.json").write_text('{"a":null,"b":null,"c":null,"x0":%s,"w_min":null,"k_min":null}' % args.x0)
        import sys; sys.exit(0)
    w = df[cols["omega2"]].astype(float, copy=False).values
    k = df[cols["kphi_min"]].astype(float, copy=False).values

    # Aggregate across seeds if present
    if "seed" in cols:
        g = df.groupby(cols["omega2"])
        agg = g[cols["kphi_min"]].apply(list).reset_index()
        w_plot = agg[cols["omega2"]].astype(float, copy=False).values
        means, los, his = [], [], []
        for arr in agg[cols["kphi_min"]].values:
            m, lo, hi = ci95(arr)
            means.append(m); los.append(lo); his.append(hi)
        k_plot = np.array(means)
        lo_plot = np.array(los); hi_plot = np.array(his)
        has_ci = True
    else:
        w_plot = w
        k_plot = k
        lo_plot = np.full_like(k_plot, np.nan, float)
        hi_plot = np.full_like(k_plot, np.nan, float)
        has_ci = False

    # Fit
    a,b,c,w_min,k_min = quad_fit(w_plot, k_plot, x0=args.x0)

    # Sort and build curve
    idx = np.argsort(w_plot)
    w_plot = w_plot[idx]; k_plot = k_plot[idx]
    lo_plot = lo_plot[idx]; hi_plot = hi_plot[idx]
    wfine = np.linspace(w_plot.min(), w_plot.max(), 600)
    kfit = a + b*(wfine - args.x0) + c*(wfine - args.x0)**2

    # Plot
    plt.figure(figsize=(7.0, 4.6))
    if has_ci and np.isfinite(lo_plot).any():
        plt.fill_between(w_plot, lo_plot, hi_plot, alpha=0.15, label="95% CI (seeds)")
    plt.plot(w_plot, k_plot, "o", ms=4, label="Boundary samples")
    plt.plot(wfine, kfit, "-", lw=2.0, label="Quadratic fit")

    # Shade lock region (above boundary)
    if args.shade_lock:
        ymax = float(np.nanmax([k_plot.max() if k_plot.size else 0.0, kfit.max() if kfit.size else 0.0]))
        plt.fill_between(wfine, kfit, ymax*1.02, alpha=0.08, label="Lock region")

    # Mark minimum
    if np.isfinite(w_min) and np.isfinite(k_min):
        plt.plot([w_min],[k_min],'s',ms=6,label="Estimated min")
        plt.annotate(fr'$(\omega_2, K_{{\phi,\min}})\approx({w_min:.3f},{k_min:.3f})$',
                     xy=(w_min,k_min), xytext=(5,10), textcoords='offset points')

    plt.xlabel(r"$\omega_2$")
    plt.ylabel(r"$K_{\phi,\min}$")
    plt.title("Phase-10 Arnold Tongue — Boundary & Fit")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    out_png = Path(f"{args.out_prefix}.png")
    out_pdf = Path(f"{args.out_prefix}.pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)

    # Fit JSON
    fitj = {
        "a": float(a), "b": float(b), "c": float(c), "x0": float(args.x0),
        "w_min": float(w_min) if np.isfinite(w_min) else None,
        "k_min": float(k_min) if np.isfinite(k_min) else None
    }
    fit_json = Path(f"{args.out_prefix}.fit.json")
    fit_json.write_text(json.dumps(fitj, indent=2))

    print(f"[phase10+] Wrote: {out_png}")
    print(f"[phase10+] Wrote: {out_pdf}")
    print(f"[phase10+] Wrote: {fit_json}")
    print(f"[phase10+] Fit: Kphi_min(ω2) = {a:.4f} + {b:.4f}(ω2-{args.x0:.3f}) + {c:.4f}(ω2-{args.x0:.3f})^2")
    if np.isfinite(w_min) and np.isfinite(k_min):
        print(f"[phase10+] Minimum at ω2 ≈ {w_min:.4f}, Kphi_min ≈ {k_min:.4f}")

if __name__ == "__main__":
    main()
