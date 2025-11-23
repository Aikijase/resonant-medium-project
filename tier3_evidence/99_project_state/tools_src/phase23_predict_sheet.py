#!/usr/bin/env python3
"""
Phase-23: Recycling – Predictions Sheet

Purpose:
  - Using Phase-21/22 outputs, compute a small set of *testables* for near-term checks:
      * ΔNeff proxy (very heuristic)
      * shift in lensing amplitude A_L proxy
      * fractional delta in growth fσ8 proxy at z={0.5,1.0,2.0}

Inputs:
  - --source-csv : outputs/phase21/recycling_source.csv
  - --fit-json   : outputs/phase22/recycling_fit.json (optional; only to echo best params)
Outputs:
  - outputs/phase23/predictions.csv
  - outputs/phase23/predictions.txt
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Phase-23: Recycling – Predictions Sheet")
    ap.add_argument("--source-csv", default="outputs/phase21/recycling_source.csv")
    ap.add_argument("--fit-json", default="outputs/phase22/recycling_fit.json")
    ap.add_argument("--outdir", default="outputs/phase23")
    args = ap.parse_args()

    print("=== Phase-23: Recycling – Predictions Sheet ===")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.source_csv):
        raise FileNotFoundError("Missing Phase-21 CSV. Run phase21_recycling_ode.py first.")

    df = pd.read_csv(args.source_csv).sort_values('z', ascending=False).reset_index(drop=True)

    best = {}
    if os.path.exists(args.fit_json):
        try:
            J = json.load(open(args.fit_json))
            best = J.get("best", {})
        except Exception:
            best = {}

    # Heuristic proxies (to be replaced by full pipeline later):
    # ΔNeff proxy ~ integral of |w_proxy+1| weighted at low z
    z = df['z'].values
    w = df['w_proxy'].values
    weight = np.exp(-z/1.5)
    dneff_proxy = float(np.trapz(np.abs(w+1.0)*weight, z))

    # Lens A_L proxy ~ relative variance of rho_de at low z (0..2)
    mask_low = (z<=2.0)
    rde = df.loc[mask_low, 'rho_de'].values
    if len(rde) > 3:
        al_proxy = float(np.var(rde/np.max(rde)))
    else:
        al_proxy = float('nan')

    # Growth proxy ~ monotonic negative correlation with rho_de slope; sample at points
    def growth_proxy(zstar):
        # approximate d ln rho_de / d ln a at zstar
        if not (z.min() <= zstar <= z.max()):
            return float('nan')
        zgrid = df['z'].values
        rdegrid = df['rho_de'].values + 1e-30
        # nearest neighbor index
        i = int(np.argmin(np.abs(zgrid - zstar)))
        i0 = max(1, i-2); i1 = min(len(zgrid)-2, i+2)
        zz = zgrid[i0:i1+1]
        rr = rdegrid[i0:i1+1]
        dln = np.gradient(np.log(rr), np.log(1+zz+1e-12))
        # map to growth: larger (more negative) slope → suppressed growth
        gp = 1.0 - 0.2*np.clip(-dln.mean(), 0, 5)
        return max(0.3, min(1.2, float(gp)))

    rows = []
    for zstar in [0.5, 1.0, 2.0]:
        rows.append({"z": zstar, "fs8_proxy": growth_proxy(zstar)})

    pred_csv = Path(args.outdir) / "predictions.csv"
    pred_txt = Path(args.outdir) / "predictions.txt"

    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    with open(pred_txt, "w") as f:
        f.write("Phase-23: Recycling – Predictions Sheet\n")
        if best:
            f.write(f"Best (Phase-22): k={best.get('k')}, alpha={best.get('alpha')}, epsilon={best.get('epsilon')}\n")
        f.write(f"ΔNeff_proxy = {dneff_proxy:.5g}\n")
        f.write(f"A_L_proxy    = {al_proxy:.5g}\n")
        for r in rows:
            f.write(f"fσ8_proxy(z={r['z']}) = {r['fs8_proxy']:.4f}\n")

    print(f"Wrote {pred_csv}")
    print(f"Wrote {pred_txt}")
    print("\nRESULT: PASS")

if __name__ == "__main__":
    main()
