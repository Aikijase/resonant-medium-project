#!/usr/bin/env python3
"""
Plot LCDM vs RESN metrics from a Thrace std_summary.json.

Usage:
  PYTHONPATH=. python tools/thrace_plot.py \
    --summary outputs/thrace_summaries/joint_baseline.std_summary.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Plot LCDM vs RESN metrics from std_summary.json")
    ap.add_argument("--summary", required=True, help="Path to *.std_summary.json")
    args = ap.parse_args()

    p = Path(args.summary)
    if not p.exists():
        sys.exit(f"[thrace-plot] summary not found: {p}")

    data = json.loads(p.read_text())
    models = data.get("models", {})
    lcdm   = models.get("lcdm", {})
    resn   = models.get("resn", {})

    # metrics to attempt
    metrics = ["chi2", "AIC", "BIC"]
    # gather values (allow missing)
    xs, lcdm_vals, resn_vals = [], [], []
    for m in metrics:
        lv = lcdm.get(m)
        rv = resn.get(m)
        if lv is not None or rv is not None:
            xs.append(m)
            lcdm_vals.append(float(lv) if lv is not None else None)
            resn_vals.append(float(rv) if rv is not None else None)

    if not xs:
        sys.exit("[thrace-plot] No plottable metrics found (need chi2/AIC/BIC in models).")

    # try to import matplotlib
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("[thrace-plot] matplotlib not installed. Install inside your venv with:")
        print("  python -m pip install matplotlib")
        return

    idx = np.arange(len(xs))
    width = 0.38

    # replace None with NaN for plotting
    lcdm_arr = np.array([float('nan') if v is None else v for v in lcdm_vals], dtype=float)
    resn_arr = np.array([float('nan') if v is None else v for v in resn_vals], dtype=float)

    fig = plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    ax.bar(idx - width/2, lcdm_arr, width, label="LCDM")
    ax.bar(idx + width/2, resn_arr, width, label="RESN")
    ax.set_xticks(idx)
    ax.set_xticklabels(xs)
    ax.set_ylabel("value")
    ax.set_title("Thrace: LCDM vs RESN metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    outdir = p.parent
    stem = p.stem.replace(".std_summary", "")
    out_png = outdir / f"{stem}.metrics.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[thrace-plot] wrote {out_png}")

if __name__ == "__main__":
    main()
