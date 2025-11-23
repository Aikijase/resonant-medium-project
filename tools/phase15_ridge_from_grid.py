#!/usr/bin/env python3
"""
Phase 15 — Ridge (max-Si) vs Centerline

1) Interpolate scattered shell (omega2,Kphi,Si) to a grid.
2) Extract ridge: for each omega2, pick Kphi that maximizes Si.
3) (Optional) Load Phase-14 centerline and compute delta = ridge - centerline.
4) Fit a quadratic to ridge: Kphi_ridge = a + b*(w2 - w0) + c*(w2 - w0)^2.
5) Save CSVs/PNGs and print stats suitable for a report card.

Outputs:
  - <prefix>_ridge.csv                   [omega2,Kphi_ridge,Si_max]
  - <prefix>_ridge.png                   [contours + ridge]
  - <prefix>_ridge_vs_centerline.png     [ridge, centerline, and ΔKphi]
  - Console: curvature, ΔKphi stats
"""

import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_shell(path):
    df = pd.read_csv(path).dropna(subset=["omega2","Kphi","sync_index"])
    if df.empty:
        print("ERROR: empty shell after NaNs", file=sys.stderr); sys.exit(1)
    return df

def make_grid(df, nw=600, nk=600, pad=0.02):
    w = df["omega2"].to_numpy(); k = df["Kphi"].to_numpy()
    wmin, wmax = float(w.min()), float(w.max())
    kmin, kmax = float(k.min()), float(k.max())
    wr = wmax - wmin; kr = kmax - kmin
    W = np.linspace(wmin - pad*wr, wmax + pad*wr, nw)
    K = np.linspace(kmin - pad*kr, kmax + pad*kr, nk)
    WW, KK = np.meshgrid(W, K)
    return WW, KK

def interp(df, WW, KK):
    w = df["omega2"].to_numpy(); k = df["Kphi"].to_numpy(); z = df["sync_index"].to_numpy()
    Z_lin = griddata((w, k), z, (WW, KK), method="linear")
    Z_near = griddata((w, k), z, (WW, KK), method="nearest")
    return np.where(np.isnan(Z_lin), Z_near, Z_lin)

def ridge_from_grid(W, K, Z):
    idx = np.argmax(Z, axis=0)                      # max along K for each omega2 column
    w_axis = W[0, :]
    k_ridge = K[idx, np.arange(K.shape[1])]
    si_max  = Z[idx, np.arange(Z.shape[1])]
    return pd.DataFrame({"omega2": w_axis, "Kphi_ridge": k_ridge, "Si_max": si_max})

def fit_quadratic(x, y):
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

def plot_ridge(W, K, Z, ridge_df, out_png):
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=160)
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 12)
    cs = ax.contour(W, K, Z, levels=levels)
    ax.plot(ridge_df["omega2"], ridge_df["Kphi_ridge"], lw=2.0, label="ridge (max Si)")
    ax.set_xlabel(r"$\omega^2$"); ax.set_ylabel(r"$K_\phi$")
    ax.set_title("Ridge over interpolated sync index field")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_ridge_vs_centerline(ridge_df, center_df, out_png):
    c = center_df.sort_values("omega2")
    r = ridge_df.sort_values("omega2")
    Kc = np.interp(r["omega2"].to_numpy(), c["omega2"].to_numpy(), c["Kphi_mid"].to_numpy())
    delta = r["Kphi_ridge"].to_numpy() - Kc
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    ax.plot(r["omega2"], Kc, lw=1.6, label="centerline")
    ax.plot(r["omega2"], r["Kphi_ridge"], lw=1.6, label="ridge")
    ax2 = ax.twinx()
    ax2.plot(r["omega2"], delta, lw=1.0, alpha=0.8, label="ΔKϕ (ridge-center)", color="tab:gray")
    ax.set_xlabel(r"$\omega^2$"); ax.set_ylabel(r"$K_\phi$")
    ax2.set_ylabel(r"$\Delta K_\phi$")
    ax.set_title("Ridge vs Centerline")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    d = pd.DataFrame({"omega2": r["omega2"], "delta": delta})
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell-csv", required=True)
    ap.add_argument("--outdir", default="outputs/phase15")
    ap.add_argument("--prefix", default="p15")
    ap.add_argument("--nw", type=int, default=600)
    ap.add_argument("--nk", type=int, default=600)
    ap.add_argument("--centerline-csv", default="")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = load_shell(args.shell_csv)
    WW, KK = make_grid(df, args.nw, args.nk)
    Z = interp(df, WW, KK)

    ridge = ridge_from_grid(WW, KK, Z)
    a,b,c,x0,R2 = fit_quadratic(ridge["omega2"], ridge["Kphi_ridge"])

    ridge_csv = os.path.join(args.outdir, f"{args.prefix}_ridge.csv")
    ridge_png = os.path.join(args.outdir, f"{args.prefix}_ridge.png")
    ridge.to_csv(ridge_csv, index=False)
    plot_ridge(WW, KK, Z, ridge, ridge_png)

    print("ridge_csv:", ridge_csv)
    print("ridge_png:", ridge_png)
    print(f"ridge_quadratic_fit: a={a:.6f}  b={b:.6f}  c={c:.6f}  (around w0={x0:.6f})  R2={R2:.4f}")
    print(f"ridge_curvature_c: {c:.6f}")

    if args.centerline_csv:
        center = pd.read_csv(args.centerline_csv)
        comp_png = os.path.join(args.outdir, f"{args.prefix}_ridge_vs_centerline.png")
        d = plot_ridge_vs_centerline(ridge, center, comp_png)
        stats = d["delta"].describe().to_dict()
        print("ridge_vs_centerline_png:", comp_png)
        print("delta_stats:", {k: float(v) for k,v in stats.items()})

if __name__ == "__main__":
    raise SystemExit(main())
