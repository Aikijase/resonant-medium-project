#!/usr/bin/env python3
"""
Phase-21: Cosmic Recycling – Source ODE (toy model)

Purpose:
  - Link BH mass-density history ρ_BH(z) to effective dark components via a simple source term.
  - Produce toy trajectories for ρ_DE(z), ρ_DM(z) and derived w(z) proxy for comparison later.

Model:
  dρ_DE/dz = - S_BH(z)
  dρ_DM/dz = + ε · S_BH(z)
  S_BH(z)  = k · [ρ_BH(z)]^α

Inputs:
  - --bh-csv: CSV with columns [z, rho_bh] (if missing, we synthesize a smooth placeholder).

Outputs:
  - outputs/phase21/recycling_source.csv         (z, rho_bh, rho_de, rho_dm, w_proxy)
  - outputs/phase21/recycling_source_report.txt  (human-readable summary)
  - outputs/phase21/recycling_source_plot.png    (diagnostic plot)

Notes:
  - This is a toy redshift-derivative model in z (not scale factor). It's sufficient for phase plumbing.
  - We clamp integrator for monotonic z-grid and basic stability; no cosmological units asserted (relative units OK).
"""
import argparse, os, sys, math, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_rho_bh(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if not {'z','rho_bh'}.issubset(set(df.columns)):
            raise ValueError("BH CSV must have columns: z, rho_bh")
        df = df.sort_values('z', ascending=False).reset_index(drop=True)
        return df
    # synthesize a smooth placeholder: rho_bh(z) rising toward z~2 then tapering
    z = np.linspace(0.0, 6.0, 241)[::-1]  # descending z for d/dz integration
    rho_bh = 1e-5 + 2e-4*np.exp(-((z-2.0)**2)/(2*0.9**2)) + 5e-6*z
    return pd.DataFrame({'z': z, 'rho_bh': rho_bh})

def integrate_toy(df, k: float, alpha: float, eps: float, rho_de0: float, rho_dm0: float):
    z = df['z'].values  # descending
    rb = df['rho_bh'].values
    n = len(z)
    rho_de = np.zeros(n); rho_dm = np.zeros(n)
    rho_de[0] = rho_de0; rho_dm[0] = rho_dm0
    # simple backward Euler in z (descending)
    for i in range(1, n):
        dz = z[i] - z[i-1]  # negative (since z decreases)
        S = k * (max(rb[i-1], 0.0)**alpha)
        rho_de[i] = rho_de[i-1] - S*dz        # minus because dz<0, S>=0 → grows forward to low-z
        rho_dm[i] = rho_dm[i-1] + eps*S*dz
        # floor to avoid negatives due to numeric quirks
        rho_de[i] = max(rho_de[i], 0.0)
        rho_dm[i] = max(rho_dm[i], 0.0)
    # crude w(z) proxy: -(1 + d ln rho_de / d ln(1+z))
    # (Not physical; just a diagnostic to align with Phase-8 shape.)
    onepz = 1.0 + z
    dln_rde = np.gradient(np.log(rho_de + 1e-30), np.log(onepz + 1e-12))
    w_proxy = -(1.0 + dln_rde)
    out = df.copy()
    out['rho_de'] = rho_de
    out['rho_dm'] = rho_dm
    out['w_proxy'] = w_proxy
    return out

def main():
    ap = argparse.ArgumentParser(description="Phase-21: Cosmic Recycling – Source ODE (toy).")
    ap.add_argument("--bh-csv", default="", help="CSV with columns z,rho_bh (optional).")
    ap.add_argument("--k", type=float, default=0.1, help="Source normalization.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Source exponent.")
    ap.add_argument("--epsilon", type=float, default=0.7, help="DM fraction ε.")
    ap.add_argument("--rho-de0", type=float, default=1.0, help="Boundary ρ_DE at highest z in grid.")
    ap.add_argument("--rho-dm0", type=float, default=0.3, help="Boundary ρ_DM at highest z in grid.")
    ap.add_argument("--outdir", default="outputs/phase21", help="Output directory.")
    args = ap.parse_args()

    print("=== Phase-21: Cosmic Recycling – Source ODE ===")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    try:
        df_bh = load_rho_bh(args.bh_csv)
        print(f"[INFO] BH curve: N={len(df_bh)}  z-range=[{df_bh['z'].min():.2f},{df_bh['z'].max():.2f}]")
        out = integrate_toy(df_bh, args.k, args.alpha, args.epsilon, args.rho_de0, args.rho_dm0)

        csv_path = Path(args.outdir) / "recycling_source.csv"
        txt_path = Path(args.outdir) / "recycling_source_report.txt"
        png_path = Path(args.outdir) / "recycling_source_plot.png"

        out.to_csv(csv_path, index=False)

        with open(txt_path, "w") as f:
            f.write("Phase-21: Cosmic Recycling – Source ODE (toy)\n")
            f.write(f"Params: k={args.k}  alpha={args.alpha}  epsilon={args.epsilon}\n")
            f.write(f"Rows: {len(out)}\n")
            f.write(f"z-range: [{out['z'].min():.3f}, {out['z'].max():.3f}]\n")
            f.write("Notes: w_proxy is a qualitative diagnostic only.\n")

        plt.figure(figsize=(8,5))
        plt.plot(out['z'], out['rho_bh'], label='rho_bh')
        plt.plot(out['z'], out['rho_de'], label='rho_de')
        plt.plot(out['z'], out['rho_dm'], label='rho_dm')
        plt.xlabel("z (descending)"); plt.ylabel("density (arb.)"); plt.title("Phase-21: Recycling Source ODE")
        plt.gca().invert_xaxis()
        plt.legend(); plt.tight_layout()
        plt.savefig(png_path, dpi=140)
        print(f"Wrote {csv_path}")
        print(f"Wrote {txt_path}")
        print(f"Wrote {png_path}")
        print("\nRESULT: PASS")

    except Exception as e:
        err_path = Path(args.outdir) / "error.txt"
        with open(err_path, "w") as f:
            f.write(str(e))
        print(f"[ERROR] {e}")
        print("RESULT: FAIL")
        sys.exit(2)

if __name__ == "__main__":
    main()
