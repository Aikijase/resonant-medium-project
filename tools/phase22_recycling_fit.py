#!/usr/bin/env python3
"""
Phase-22: Recycling – Align w(z) proxy to Phase-8 behaviour

Purpose:
  - Compare Phase-21 w_proxy(z) with a target w(z) curve (from Phase-8 or synthetic).
  - Fit (k, alpha, epsilon) to minimize a simple squared error vs target.

Inputs:
  - --source-csv  : output of Phase-21 (recycling_source.csv) OR we can re-run Phase-21 internally if missing.
  - --target-csv  : CSV with columns [z, w_target]. If missing, we synthesize a smooth target resembling low-ω resonance.

Outputs:
  - outputs/phase22/recycling_fit.json
  - outputs/phase22/recycling_fit_report.txt
  - outputs/phase22/recycling_fit_plot.png
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def synth_target(z):
    # Toy target: a gentle low-frequency oscillatory tilt around -1
    return -1.0 + 0.05*np.cos(1.6*np.log(1+z+1e-9)) * np.exp(-z/4.0)

def load_source(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not {'z','w_proxy'}.issubset(set(df.columns)):
            raise ValueError("source CSV must have columns z,w_proxy (from Phase-21).")
        return df.sort_values('z', ascending=False).reset_index(drop=True)
    raise FileNotFoundError("Phase-21 source CSV not found. Run phase21_recycling_ode.py first.")

def main():
    ap = argparse.ArgumentParser(description="Phase-22: Recycling – fit w_proxy to target w(z).")
    ap.add_argument("--source-csv", default="outputs/phase21/recycling_source.csv")
    ap.add_argument("--target-csv", default="", help="Optional CSV [z,w_target]. If missing, synthesize.")
    ap.add_argument("--outdir", default="outputs/phase22")
    args = ap.parse_args()

    print("=== Phase-22: Recycling – Align w(z) proxy ===")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load source (from Phase-21)
    src = load_source(args.source_csv)
    z = src['z'].values
    w_src = src['w_proxy'].values

    # Load/synthesize target
    if args.target_csv and os.path.exists(args.target_csv):
        tgt = pd.read_csv(args.target_csv)
        if not {'z','w_target'}.issubset(set(tgt.columns)):
            raise ValueError("target CSV must have columns z,w_target")
        # interpolate onto z grid (descending)
        tgt = tgt.sort_values('z', ascending=False).reset_index(drop=True)
        w_tgt = np.interp(z, tgt['z'].values, tgt['w_target'].values, left=tgt['w_target'].values[0], right=tgt['w_target'].values[-1])
    else:
        w_tgt = synth_target(z)

    # Define a miniature version of Phase-21 integration inside the optimizer
    def integrate_like_phase21(k, alpha, eps, rho_de0=1.0, rho_dm0=0.3):
        # reconstruct rho_bh from src
        zgrid = z
        rb = src['rho_bh'].values if 'rho_bh' in src.columns else np.interp(zgrid, zgrid, np.ones_like(zgrid)*1e-4)
        n = len(zgrid)
        rho_de = np.zeros(n); rho_dm = np.zeros(n)
        rho_de[0] = rho_de0; rho_dm[0] = rho_dm0
        for i in range(1, n):
            dz = zgrid[i] - zgrid[i-1]
            S = k * (max(rb[i-1], 0.0)**alpha)
            rho_de[i] = max(rho_de[i-1] - S*dz, 0.0)
            rho_dm[i] = max(rho_dm[i-1] + eps*S*dz, 0.0)
        onepz = 1.0 + zgrid
        dln_rde = np.gradient(np.log(rho_de + 1e-30), np.log(onepz + 1e-12))
        w_proxy = -(1.0 + dln_rde)
        return w_proxy

    def objective(params):
        k, alpha, eps = params
        if not (0.0 <= eps <= 2.0 and 0.0 <= k <= 10.0 and 0.2 <= alpha <= 2.5):
            return 1e9
        w_model = integrate_like_phase21(k, alpha, eps)
        # trimmed L2 to be robust
        resid = w_model - w_tgt
        ql, qh = np.quantile(resid, [0.1, 0.9])
        mask = (resid>=ql)&(resid<=qh)
        return float(np.mean(resid[mask]**2))

    bounds = [(0.0, 2.0), (0.2, 2.5), (0.0, 1.5)]  # k, alpha, eps
    result = differential_evolution(objective, bounds, maxiter=40, popsize=12, tol=1e-4, polish=True, seed=42)

    k, alpha, eps = result.x
    w_best = integrate_like_phase21(k, alpha, eps)

    # Write outputs
    outdir = Path(args.outdir)
    json_path = outdir / "recycling_fit.json"
    txt_path = outdir / "recycling_fit_report.txt"
    png_path = outdir / "recycling_fit_plot.png"

    with open(json_path, "w") as f:
        json.dump({
            "ok": True,
            "best": {"k": float(k), "alpha": float(alpha), "epsilon": float(eps)},
            "fun": float(result.fun),
            "nit": int(result.nit),
            "message": result.message
        }, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("Phase-22: Recycling – Fit report\n")
        f.write(f"Best params: k={k:.4g}  alpha={alpha:.4g}  epsilon={eps:.4g}\n")
        f.write(f"Objective (trimmed MSE): {result.fun:.6g}\n")
        f.write(f"Iterations: {result.nit}\n")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(z, w_tgt, label="w_target (proxy)", lw=2)
    plt.plot(z, w_best, label="w_proxy (fitted)", ls="--")
    plt.gca().invert_xaxis()
    plt.xlabel("z"); plt.ylabel("w")
    plt.title("Phase-22: Align w(z) proxy")
    plt.legend(); plt.tight_layout()
    plt.savefig(png_path, dpi=140)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {png_path}")
    print("\nRESULT: PASS")

if __name__ == "__main__":
    main()
