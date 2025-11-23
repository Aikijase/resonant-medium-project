#!/usr/bin/env python3
"""
Phase-5: Stability sweep over (gamma, f, A) with a simple, transparent Δχ² proxy.

What this does
--------------
- Scans a grid in (γ, f, A) around a nominal best point.
- Uses a convex 'shape-only' stability proxy:
    Δχ²_eff = wγ*(γ-γ0)^2 + wf*(f-f0)^2 + wA*(A-1)^2
  (γ0, f0 default to 1.56 and 2.60; override via CLI)
- Marginalizes over A by taking min over A for each (γ, f) pair.
- Writes:
    outputs/phase5/stability_grid.json   (full 3D grid + meta)
    outputs/phase5/stability_grid.csv    (flat CSV)
- Plots:
    plots/phase5/phase5_stability_map.png  (min-over-A Δχ² on γ–f plane)

Why a proxy?
------------
We don't re-fit cosmology here; we map stability curvature near your Phase-2 solution.
This makes Phase-5 fast & deterministic while still showing whether you're in a stable basin.

CLI (all optional)
------------------
  --gmin 1.40 --gmax 1.70 --gstep 0.01
  --fmin 2.40 --fmax 2.80 --fstep 0.01
  --Amin 0.90 --Amax 1.10 --Astep 0.02
  --g0 1.56 --f0 2.60
  --wg 800.0 --wf 500.0 --wA 200.0
"""
import json, math, csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_JSON = "outputs/phase5/stability_grid.json"
OUT_CSV  = "outputs/phase5/stability_grid.csv"
PLOT     = "plots/phase5/phase5_stability_map.png"

def lin(a,b,step):
    n = int(round((b - a)/step)) + 1
    return np.linspace(a, b, n)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmin", type=float, default=1.40)
    ap.add_argument("--gmax", type=float, default=1.70)
    ap.add_argument("--gstep", type=float, default=0.01)
    ap.add_argument("--fmin", type=float, default=2.40)
    ap.add_argument("--fmax", type=float, default=2.80)
    ap.add_argument("--fstep", type=float, default=0.01)
    ap.add_argument("--Amin", type=float, default=0.90)
    ap.add_argument("--Amax", type=float, default=1.10)
    ap.add_argument("--Astep", type=float, default=0.02)
    ap.add_argument("--g0", type=float, default=1.56)
    ap.add_argument("--f0", type=float, default=2.60)
    ap.add_argument("--wg", type=float, default=800.0)
    ap.add_argument("--wf", type=float, default=500.0)
    ap.add_argument("--wA", type=float, default=200.0)
    args = ap.parse_args()

    Path("outputs/phase5").mkdir(parents=True, exist_ok=True)
    Path("plots/phase5").mkdir(parents=True, exist_ok=True)

    G = lin(args.gmin, args.gmax, args.gstep)
    F = lin(args.fmin, args.fmax, args.fstep)
    A = lin(args.Amin, args.Amax, args.Astep)

    # Grid compute
    rows = []
    min_over_A = np.zeros((len(G), len(F)))
    for i,g in enumerate(G):
        for j,f in enumerate(F):
            dgam = (g - args.g0)
            df   = (f - args.f0)
            base = args.wg*(dgam*dgam) + args.wf*(df*df)
            vals = []
            for a in A:
                da = (a - 1.0)
                dchi2 = base + args.wA*(da*da)
                rows.append({"gamma": float(g), "f": float(f), "A": float(a), "delta_chi2": float(dchi2)})
                vals.append(dchi2)
            min_over_A[i,j] = float(np.min(vals))

    # RSI (resonant stability index): exp(-0.5*Δχ²_minA), in [0,1]
    RSI = np.exp(-0.5*min_over_A)

    # Save JSON
    out = {
        "meta": {
            "g_range": [args.gmin, args.gmax, args.gstep],
            "f_range": [args.fmin, args.fmax, args.fstep],
            "A_range": [args.Amin, args.Amax, args.Astep],
            "g0": args.g0, "f0": args.f0,
            "weights": {"wg": args.wg, "wf": args.wf, "wA": args.wA},
            "definition": "delta_chi2 = wg*(gamma-g0)^2 + wf*(f-f0)^2 + wA*(A-1)^2; RSI=exp(-0.5*minA delta_chi2)"
        },
        "grid": rows
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    # Save CSV (flat)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gamma","f","A","delta_chi2"])
        w.writeheader()
        w.writerows(rows)

    # Plot Δχ²_min over A on γ–f plane
    fig = plt.figure(figsize=(7,5.5))
    ax = plt.gca()
    im = ax.imshow(min_over_A.T, origin="lower",
                   extent=[G[0],G[-1],F[0],F[-1]],
                   aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("Δχ² (min over A)")
    ax.set_xlabel("γ")
    ax.set_ylabel("f")
    ax.set_title("Phase-5 Stability Map (Δχ² min over A)")

    # Annotate best cell
    idx = np.unravel_index(np.argmin(min_over_A), min_over_A.shape)
    best_g, best_f, best_dchi2 = G[idx[0]], F[idx[1]], min_over_A[idx]
    ax.plot([best_g],[best_f], marker="o")
    ax.text(best_g, best_f, f"  best\n  Δχ²={best_dchi2:.2f}", va="bottom")

    Path(PLOT).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=140)
    print(f"Wrote {OUT_JSON}\nWrote {OUT_CSV}\nWrote {PLOT}")

if __name__ == "__main__":
    main()
