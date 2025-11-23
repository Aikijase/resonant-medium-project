#!/usr/bin/env python3
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def quad_vertex(x, y):
    # fit y = a + b x + c x^2, return vertex x* = -b/(2c) if c!=0 else argmax x
    X = np.column_stack([np.ones_like(x), x, x**2])
    (a,b,c), *_ = np.linalg.lstsq(X, y, rcond=None)
    if abs(c) < 1e-12:
        return x[np.argmax(y)], float(a), float(b), float(c)
    return float(-b/(2*c)), float(a), float(b), float(c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell-csv", required=True)
    ap.add_argument("--outdir", default="outputs/phase15")
    ap.add_argument("--prefix", default="p15_ref")
    ap.add_argument("--nw", type=int, default=600)
    ap.add_argument("--nk", type=int, default=600)
    ap.add_argument("--window", type=int, default=5, help="odd window along Kphi for local quad fit (>=3)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.shell_csv).dropna(subset=["omega2","Kphi","sync_index"])
    w, k, z = df["omega2"].values, df["Kphi"].values, df["sync_index"].values

    # grid
    W_lin = np.linspace(w.min(), w.max(), args.nw)
    K_lin = np.linspace(k.min(), k.max(), args.nk)
    WW, KK = np.meshgrid(W_lin, K_lin)
    Z_lin = griddata((w,k), z, (WW,KK), method="linear")
    Z_near= griddata((w,k), z, (WW,KK), method="nearest")
    Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)

    half = max(1, args.window//2)
    ridge_w, ridge_k, ridge_si = [], [], []

    for j in range(WW.shape[1]):  # per omega column
        col = Z[:, j]
        imax = int(np.nanargmax(col))
        i0, i1 = max(0, imax-half), min(Z.shape[0]-1, imax+half)
        idx = np.arange(i0, i1+1)
        kk = K_lin[idx]; yy = col[idx]
        # local quad in Kphi
        k_star, a,b,c = quad_vertex(kk, yy)
        k_star = np.clip(k_star, K_lin[0], K_lin[-1])
        # sample Si at k_star by 1D interp
        si_star = np.interp(k_star, K_lin, col)
        ridge_w.append(W_lin[j]); ridge_k.append(k_star); ridge_si.append(si_star)

    ridge = pd.DataFrame({"omega2": ridge_w, "Kphi_ridge_ref": ridge_k, "Si_max_ref": ridge_si})

    # fit curvature on refined ridge
    x = ridge["omega2"].values; y = ridge["Kphi_ridge_ref"].values
    x0 = 0.5*(x.min()+x.max())
    X = np.column_stack([np.ones_like(x), (x-x0), (x-x0)**2])
    (a,b,c), *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ np.array([a,b,c])
    R2 = 1.0 - np.sum((y-yhat)**2)/max(np.sum((y-y.mean())**2), 1e-12)

    ridge_csv = os.path.join(args.outdir, f"{args.prefix}_ridge_refined.csv")
    ridge_png = os.path.join(args.outdir, f"{args.prefix}_ridge_refined.png")
    ridge.to_csv(ridge_csv, index=False)

    # plot
    fig, ax = plt.subplots(figsize=(7.6,5.2), dpi=160)
    cs = ax.contour(WW, KK, Z, levels=10)
    ax.plot(ridge["omega2"], ridge["Kphi_ridge_ref"], lw=2.0, label="ridge refined")
    ax.set_xlabel(r"$\omega^2$"); ax.set_ylabel(r"$K_\phi$")
    ax.set_title("Refined ridge (local quadratic peak)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(ridge_png); plt.close(fig)

    print("ridge_refined_csv:", ridge_csv)
    print("ridge_refined_png:", ridge_png)
    print(f"ridge_refined_fit: a={a:.6f} b={b:.6f} c={c:.6f} (around w0={x0:.6f}) R2={R2:.4f}")

if __name__ == "__main__":
    raise SystemExit(main())
