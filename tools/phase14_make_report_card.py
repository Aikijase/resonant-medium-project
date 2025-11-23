#!/usr/bin/env python3
"""
Reads a phase14_batch_widths JSON, loads each width CSV, and produces:
  - A combined width-vs-omega2 figure (all pairs)
  - A Markdown report card with a compact table + paragraph
  - A LaTeX snippet (optional) you can paste into Overleaf

Usage:
  python3 tools/phase14_make_report_card.py \
    --summary-json outputs/phase14/p14_edge_batch_summary_YYYYMMDD-HHMMSS.json \
    --outdir outputs/phase14 --prefix p14_edge

Notes:
  - Tiny negative widths are clamped to 0 for plotting + stats polish.
  - Uses only files already produced by your existing tools.
"""
import argparse, json, os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clamp_nonneg(x):
    x = np.asarray(x, float)
    x[x < 0] = 0.0
    return x

def load_rows(path_json):
    with open(path_json, "r") as f:
        rows = json.load(f)
    # keep only successful
    return [r for r in rows if r.get("status") == "ok" and r.get("width_csv")]

def summarize_width_csv(width_csv):
    df = pd.read_csv(width_csv)
    w2 = df["omega2"].values
    width = clamp_nonneg(df["width"].values)
    # basic stats
    stats = {
        "omega2_min": float(np.nanmin(w2)),
        "omega2_max": float(np.nanmax(w2)),
        "mean_width": float(np.nanmean(width)),
        "median_width": float(np.nanmedian(width)),
        "q25_width": float(np.nanpercentile(width, 25)),
        "q75_width": float(np.nanpercentile(width, 75)),
        "count": int(width.size),
    }
    # area (trapz) recomputed after clamp for display neatness
    stats["A_trapz_clean"] = float(np.trapz(width, w2))
    return stats

def make_plot(rows, out_png):
    plt.figure(figsize=(8.0, 4.6), dpi=160)
    for r in rows:
        df = pd.read_csv(r["width_csv"])
        w2 = df["omega2"].values
        width = clamp_nonneg(df["width"].values)
        label = f"Si {r['low']:.2f} → {r['high']:.2f}"
        plt.plot(w2, width, lw=1.6, label=label)
    plt.xlabel(r'$\omega^2$'); plt.ylabel(r'Width in $K_\phi$')
    plt.title('Lock-tongue width vs $\\omega^2$ (multiple Si bands)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def write_markdown(rows, stats_by_csv, out_md):
    lines = []
    lines.append("# Phase 14 — Tongue Width & Area Summary\n")
    lines.append("| Pair (Si low→high) | Merge | ω² span | mean width | A_trapz | A_mask | Notes |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        st = stats_by_csv[r["width_csv"]]
        span = f"{st['omega2_min']:.3f}→{st['omega2_max']:.3f}"
        lines.append(
            f"| {r['low']:.2f}→{r['high']:.2f} | {r.get('merge_mode','?')} "
            f"| {span} | {r.get('mean_width',float('nan')):.4f} "
            f"| {r.get('A_trapz',float('nan')):.6f} | {r.get('A_mask',float('nan')):.6f} | "
            f"A_trapz(clean): {st['A_trapz_clean']:.6f} |"
        )
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

def write_latex(rows, stats_by_csv, out_tex):
    lines = []
    lines.append(r"% Phase 14 — Tongue Width & Area Summary")
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Pair & Merge & $\omega^2$ span & $\overline{w}$ & $A_{\mathrm{trapz}}$ & $A_{\mathrm{mask}}$ \\")
    lines.append(r"\midrule")
    for r in rows:
        st = stats_by_csv[r["width_csv"]]
        span = f"{st['omega2_min']:.3f}--{st['omega2_max']:.3f}"
        lines.append(
            f"Si {r['low']:.2f}→{r['high']:.2f} & {r.get('merge_mode','?')} & {span} & "
            f"{r.get('mean_width',float('nan')):.4f} & {r.get('A_trapz',float('nan')):.6f} & "
            f"{r.get('A_mask',float('nan')):.6f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--outdir", default="outputs/phase14")
    ap.add_argument("--prefix", default="p14_edge")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.summary_json)

    # compute refined stats per width CSV
    stats_by_csv = {}
    for r in rows:
        stats_by_csv[r["width_csv"]] = summarize_width_csv(r["width_csv"])

    # combined plot
    combo_png = os.path.join(args.outdir, f"{args.prefix}_width_combo.png")
    make_plot(rows, combo_png)

    # markdown + latex
    md_path  = os.path.join(args.outdir, f"{args.prefix}_report_card.md")
    tex_path = os.path.join(args.outdir, f"{args.prefix}_report_table.tex")
    write_markdown(rows, stats_by_csv, md_path)
    write_latex(rows, stats_by_csv, tex_path)

    print("wrote:", combo_png)
    print("wrote:", md_path)
    print("wrote:", tex_path)

if __name__ == "__main__":
    raise SystemExit(main())
