#!/usr/bin/env python3
"""
Compare Phase 9 URO runs.

Input CSV columns (from phase9_unified_operator_sim.py):
t, x, v, x_star, err, beta_eff, energy

Outputs:
- prints a table of metrics for each input CSV
- writes outputs/phase9/metrics_summary.csv
"""
import argparse, os, csv, math, sys
from collections import OrderedDict
import numpy as np

def load_csv(path):
    cols = {"t": [], "x": [], "v": [], "x_star": [], "err": [], "beta_eff": [], "energy": []}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in cols:
                cols[k].append(float(row[k]))
    for k in cols:
        cols[k] = np.asarray(cols[k], dtype=float)
    return cols

def metrics_for_run(path):
    d = load_csv(path)
    n = len(d["t"])
    mid = n // 2
    sl = slice(mid, None)
    # Basic metrics
    p2p = float(np.max(d["x"][sl]) - np.min(d["x"][sl]))
    rms_err = float(np.sqrt(np.mean(d["err"][sl]**2)))
    mean_beta = float(np.mean(d["beta_eff"][sl]))
    mean_E = float(np.mean(d["energy"][sl]))
    # Dominant frequency via simple FFT on second half
    t = d["t"][sl]
    x = d["x"][sl]
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        dom_f = float("nan")
    else:
        X = np.fft.rfft(x - np.mean(x))
        freqs = np.fft.rfftfreq(len(x), d=dt)
        idx = np.argmax(np.abs(X)[1:]) + 1 if len(X) > 1 else 0
        dom_f = float(freqs[idx]) if len(freqs) else float("nan")
    return {
        "file": path,
        "peak_to_peak": p2p,
        "rms_err": rms_err,
        "mean_beta_eff": mean_beta,
        "mean_energy": mean_E,
        "dominant_freq_hz": dom_f,
    }

def main():
    ap = argparse.ArgumentParser(description="Compare Phase 9 URO CSV outputs")
    ap.add_argument("csvs", nargs="+", help="One or more CSV paths (e.g., outputs/phase9/*_run.csv)")
    ap.add_argument("--out", default="outputs/phase9/metrics_summary.csv", help="Where to save summary CSV")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = [metrics_for_run(p) for p in args.csvs]

    # Pretty print
    def short(p):
        return os.path.splitext(os.path.basename(p))[0]
    header = ["run", "peak_to_peak", "rms_err", "mean_beta_eff", "mean_energy", "dominant_freq_hz"]
    print("\nPhase 9 — Metrics Summary\n")
    print("{:<28} {:>12} {:>12} {:>14} {:>14} {:>18}".format(*header))
    for r in rows:
        print("{:<28} {:>12.6g} {:>12.6g} {:>14.6g} {:>14.6g} {:>18.6g}".format(
            short(r["file"]), r["peak_to_peak"], r["rms_err"], r["mean_beta_eff"], r["mean_energy"], r["dominant_freq_hz"]
        ))

    # Save CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([short(r["file"]), r["peak_to_peak"], r["rms_err"], r["mean_beta_eff"], r["mean_energy"], r["dominant_freq_hz"]])
    print(f"\n[phase9] Wrote summary: {args.out}")

if __name__ == "__main__":
    main()
