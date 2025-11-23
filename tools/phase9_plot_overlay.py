#!/usr/bin/env python3
"""
Overlay plots for Phase 9 URO runs.
Inputs: CSVs from phase9_unified_operator_sim*.py
Outputs: outputs/phase9/overlay_timeseries.png, overlay_phase.png
"""
import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(path):
    cols = {"t": [], "x": [], "v": [], "x_star": []}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in cols:
                cols[k].append(float(row[k]))
    for k in cols:
        cols[k] = np.asarray(cols[k], dtype=float)
    return cols

def short(p): 
    return os.path.splitext(os.path.basename(p))[0]

def main():
    ap = argparse.ArgumentParser(description="Overlay plots for URO runs")
    ap.add_argument("csvs", nargs="+", help="CSV paths")
    ap.add_argument("--outdir", default="outputs/phase9", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Time series overlay
    plt.figure(figsize=(10, 5))
    for p in args.csvs:
        d = load_csv(p)
        plt.plot(d["t"], d["x"], label=short(p))
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("URO Time Series — Overlay")
    plt.legend(loc="best")
    plt.tight_layout()
    ts_path = os.path.join(args.outdir, "overlay_timeseries.png")
    plt.savefig(ts_path, dpi=150)
    plt.close()
    print(f"[phase9] Saved: {ts_path}")

    # Phase portrait overlay
    plt.figure(figsize=(6, 6))
    for p in args.csvs:
        d = load_csv(p)
        plt.plot(d["x"], d["v"], label=short(p))
    plt.xlabel("x")
    plt.ylabel("v = ẋ")
    plt.title("URO Phase Portrait — Overlay")
    plt.legend(loc="best")
    plt.tight_layout()
    ph_path = os.path.join(args.outdir, "overlay_phase.png")
    plt.savefig(ph_path, dpi=150)
    plt.close()
    print(f"[phase9] Saved: {ph_path}")

if __name__ == "__main__":
    main()
