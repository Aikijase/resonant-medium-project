#!/usr/bin/env python3
"""
Phase 9 — Beta Sweep Tool
Author: J. Watts + GPT-5
Purpose: sweep mean_beta across input runs to map amplitude–error tradeoff.

Usage:
  env PYTHONPATH=. python3 tools/phase9_beta_sweep.py \
    --inputs outputs/phase9/physical_run.csv outputs/phase9/psychological_run_noscipy.csv outputs/phase9/learning_run_noscipy.csv \
    --beta-grid 0.02 0.80 40 \
    --out-prefix outputs/phase9/beta_sweep
"""

import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, os, json

def analyze_run(df):
    vals = df.select_dtypes("number").values.flatten()
    vals = vals[~np.isnan(vals)]
    peak_to_peak = np.ptp(vals)
    rms_err = np.sqrt(np.mean((vals - np.mean(vals)) ** 2))
    mean_beta = np.mean(np.gradient(vals))
    return dict(peak_to_peak=peak_to_peak, rms_err=rms_err, mean_beta=mean_beta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--beta-grid", nargs=3, type=float, metavar=("BMIN","BMAX","NSTEPS"), required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    bmin,bmax,nsteps = args.beta_grid
    betas = np.linspace(bmin,bmax,int(nsteps))
    results = []

    for path in args.inputs:
        df = pd.read_csv(path)
        name = os.path.splitext(os.path.basename(path))[0]
        base = analyze_run(df)
        for beta in betas:
            # simple simulated sweep — scale amplitude and error inversely by beta
            amp = base["peak_to_peak"] * (1 - 0.5*beta)
            err = base["rms_err"] * (1 + 2*beta)
            results.append(dict(run=name, beta=beta, amplitude=amp, error=err))

    out_csv = args.out_prefix + "_table.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

    # Plot amplitude–error frontier
    plt.figure(figsize=(7,5))
    for run in sorted(set(r["run"] for r in results)):
        sub = [r for r in results if r["run"]==run]
        plt.plot([r["error"] for r in sub], [r["amplitude"] for r in sub], label=run)
    plt.xlabel("RMS Error (proxy for precision)")
    plt.ylabel("Peak-to-Peak Amplitude (proxy for energy)")
    plt.title("Phase 9 — Amplitude–Error Frontier vs β")
    plt.legend(); plt.grid(True, alpha=0.4)
    out_png = args.out_prefix + "_frontier.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=200)

    summary = dict(
        beta_grid=list(betas),
        output_csv=out_csv,
        output_plot=out_png,
        runs=[os.path.basename(p) for p in args.inputs],
    )
    out_json = args.out_prefix + "_manifest.json"
    json.dump(summary, open(out_json,"w"), indent=2)
    print(f"[phase9] Saved: {out_csv}")
    print(f"[phase9] Saved: {out_png}")
    print(f"[phase9] Saved: {out_json}")

if __name__ == "__main__":
    main()
