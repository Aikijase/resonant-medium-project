#!/usr/bin/env python3
"""
Analyze Thrace multistart results:
- Reads outputs/thrace_multistart/leaderboard.(csv|jsonl)
- Prints top-K runs (by canonical χ²)
- Writes a small JSON of the best params
- Plots: χ² vs alpha/gamma/beta_de/beta_dm (log-x) and a coarse α–γ heatmap

Usage:
  PYTHONPATH=. python tools/thrace_ms_analyze.py --top 10
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LB_DIR = ROOT / "outputs" / "thrace_multistart"
LB_CSV = LB_DIR / "leaderboard.csv"
LB_JSONL = LB_DIR / "leaderboard.jsonl"
OUT_DIR = LB_DIR / "analysis"

FIELDS_NUM = ["idx","alpha","gamma","beta_de","beta_dm","chi2","AIC","BIC","dChi2","dAIC","dBIC","chi2_resn","chi2_lcdm","N"]

def load_leaderboard():
    rows = []
    if LB_CSV.exists():
        with LB_CSV.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                rr = {}
                for k,v in r.items():
                    if k in FIELDS_NUM and (v is not None and v != ""):
                        try: rr[k] = float(v)
                        except Exception: rr[k] = None
                    else:
                        rr[k] = v
                rows.append(rr)
    elif LB_JSONL.exists():
        with LB_JSONL.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rows.append(r)
    else:
        sys.exit(f"[ms-analyze] no leaderboard at {LB_CSV} or {LB_JSONL}")
    # filter rows with numeric chi2
    rows = [r for r in rows if isinstance(r.get("chi2"), (int,float))]
    rows.sort(key=lambda r: r["chi2"])
    return rows

def fmt(x, nd=6):
    if x is None: return "—"
    try:
        x = float(x)
        return f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.{nd}e}"
    except Exception:
        return str(x)

def print_top(rows, k):
    k = min(k, len(rows))
    print(f"=== Top {k} by χ² (lower is better) ===")
    for i in range(k):
        r = rows[i]
        print(f"#{i+1:02d}  χ²={fmt(r['chi2'])}  "
              f"α={fmt(r['alpha'])}  γ={fmt(r['gamma'])}  "
              f"β_de={fmt(r['beta_de'])}  β_dm={fmt(r['beta_dm'])}  "
              f"dir={r.get('run_dir','?')}")
    print()

def write_best_params(rows):
    best = rows[0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "best_params.json"
    best_small = {
        "alpha": best["alpha"], "gamma": best["gamma"],
        "beta_de": best["beta_de"], "beta_dm": best["beta_dm"],
        "chi2": best["chi2"], "run_dir": best.get("run_dir"),
        "summary_path": best.get("summary_path")
    }
    p.write_text(json.dumps(best_small, indent=2))
    print(f"[ms-analyze] wrote {p}")

def scatter_param(ax, xs, ys, xlabel):
    ax.scatter(xs, ys, s=24, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("χ²")
    ax.grid(True, alpha=0.3)

def heatmap_alpha_gamma(rows):
    # coarse binning in log10 space
    a = np.array([r["alpha"] for r in rows], dtype=float)
    g = np.array([r["gamma"] for r in rows], dtype=float)
    y = np.array([r["chi2"] for r in rows], dtype=float)

    # define bins
    bins = 24
    ax = np.log10(a); gx = np.log10(g)
    ax_edges = np.linspace(ax.min(), ax.max(), bins+1)
    gx_edges = np.linspace(gx.min(), gx.max(), bins+1)

    grid = np.full((bins, bins), np.nan)
    # bin by min chi2 in each cell
    for i in range(bins):
        for j in range(bins):
            mask = (ax >= ax_edges[i]) & (ax < ax_edges[i+1]) & (gx >= gx_edges[j]) & (gx < gx_edges[j+1])
            if np.any(mask):
                grid[j, i] = np.nanmin(y[mask])

    fig = plt.figure(figsize=(6,5))
    im = plt.imshow(grid, origin="lower",
                    extent=[10**ax_edges[0], 10**ax_edges[-1], 10**gx_edges[0], 10**gx_edges[-1]],
                    aspect="auto")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("alpha"); plt.ylabel("gamma"); plt.title("χ² (min/bin) over α–γ")
    cbar = plt.colorbar(im)
    cbar.set_label("χ²")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "heatmap_alpha_gamma.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
    print(f"[ms-analyze] wrote {out}")

def main():
    ap = argparse.ArgumentParser(description="Analyze multistart leaderboard and plot metrics.")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = load_leaderboard()
    if not rows:
        sys.exit("[ms-analyze] no rows with numeric χ²")

    print_top(rows, args.top)
    write_best_params(rows)

    # Plots
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # χ² vs single parameters
    pairs = [
        ("alpha",  [r["alpha"] for r in rows]),
        ("gamma",  [r["gamma"] for r in rows]),
        ("beta_de",[r["beta_de"] for r in rows]),
        ("beta_dm",[r["beta_dm"] for r in rows]),
    ]
    chi2s = [r["chi2"] for r in rows]
    for name, xs in pairs:
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        scatter_param(ax, np.array(xs, dtype=float), np.array(chi2s, dtype=float), name)
        out = OUT_DIR / f"scatter_{name}_chi2.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[ms-analyze] wrote {out}")

    # α–γ heatmap
    heatmap_alpha_gamma(rows)

if __name__ == "__main__":
    main()
