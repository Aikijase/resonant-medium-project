#!/usr/bin/env python3
"""
Phase-9: Regime Trends
Links Phase-8 metrics (Δχ²) to resonant parameters τ, κ, A, Q.
Plots simple bar and trendline summaries.

Usage:
  python3 tools/phase9_regime_trends.py outputs/phase8/phase8_metrics.csv
"""
import sys, csv, json, os
import matplotlib.pyplot as plt

def main(path):
    rows = list(csv.DictReader(open(path)))
    base = float(rows[0]["metric"])
    x, y, labels = [], [], []
    params = {"tau":0.3, "kappa":-0.02, "A":1.0}  # baseline physics anchors

    for r in rows:
        stem = os.path.basename(r["stem"])
        if not r["metric"]: continue
        val = float(r["metric"]) - base
        labels.append(stem.replace("joint_",""))
        y.append(val)
        x.append(len(y))

    plt.figure(figsize=(6,4))
    plt.bar(labels, y, color="lightsteelblue", edgecolor="black")
    plt.axhline(0, color="gray", lw=1)
    plt.title("Phase-9 Resonant Fit Δχ² vs Parameter Locks")
    plt.ylabel("Δχ² relative to baseline")
    plt.tight_layout()
    out="outputs/phase9/delta_bar.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200)
    print("Wrote", out)

    # optional JSON trend for downstream regression
    out_json="outputs/phase9/regime_trends.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump({"baseline_metric":base,
               "delta_metrics":dict(zip(labels,y)),
               "params":params}, open(out_json,"w"), indent=2)
    print("Wrote", out_json)

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python3 tools/phase9_regime_trends.py <phase8_metrics.csv>")
        sys.exit(2)
    main(sys.argv[1])
