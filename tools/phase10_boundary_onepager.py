#!/usr/bin/env python3
import sys, csv, textwrap, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Inputs (change if you used different names)
MAP_PNG   = "outputs/phase10/nerdline_seeds_fair_with_contour.png"
FIT_PNG   = "outputs/phase10/boundary_curve_fit.png"
CONTOUR_CSV = "outputs/phase10/nerdline_seeds_fair_contour.csv"
OUT_PDF   = "outputs/phase10/phase10_boundary_onepager.pdf"

def load_stats(csv_path):
    ws, ks = [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["Kphi_min_at_thr"] and row["Kphi_min_at_thr"] != "NA":
                ws.append(float(row["omega2"]))
                ks.append(float(row["Kphi_min_at_thr"]))
    if not ws:
        return None
    return {
        "w_min": min(ws), "w_max": max(ws),
        "k_min": min(ks), "k_max": max(ks),
        "k_med": float(np.median(ks)), "k_mean": float(np.mean(ks)),
        "n": len(ws)
    }

def main():
    stats = load_stats(CONTOUR_CSV)
    # Page
    plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
    gs = GridSpec(6, 1, height_ratios=[0.7, 2.6, 2.1, 0.6, 0.6, 0.4])

    # Title
    ax0 = plt.subplot(gs[0, 0])
    ax0.axis("off")
    title = "Phase 10 — Neuron Line: Phase-Lock Boundary (Sync ≥ 0.95)"
    subtitle = "Heatmap+Contour • Quadratic Fit • ω₂-span and Kφ_min stats"
    ax0.text(0, 0.65, title, ha="left", va="center", fontsize=16, weight="bold")
    ax0.text(0, 0.15, subtitle, ha="left", va="center", fontsize=10)

    # Heatmap + contour
    ax1 = plt.subplot(gs[1:3, 0])
    ax1.axis("off")
    try:
        img = plt.imread(MAP_PNG)
        ax1.imshow(img)
    except Exception as e:
        ax1.text(0.5, 0.5, f"(Could not load map image)\n{e}", ha="center", va="center")

    # Fit plot
    ax2 = plt.subplot(gs[3:5, 0])
    ax2.axis("off")
    try:
        img2 = plt.imread(FIT_PNG)
        ax2.imshow(img2)
    except Exception as e:
        ax2.text(0.5, 0.5, f"(Could not load fit image)\n{e}", ha="center", va="center")

    # Stats / notes
    ax3 = plt.subplot(gs[5, 0])
    ax3.axis("off")
    notes = [
        "Params: preset=neuron, kv=0.10, kx=0.20, eps=0.06, noise=0.01, steps=20000, burn_in=300.",
        "Method: grid heatmap (banded) → ≥0.95 contour → quadratic fit around center.",
        "Interpretation: island spans ~2.70–2.90; minimum Kφ needed at ω₂≈2.788 is ~0.337."
    ]
    if stats:
        sline = (f"ω₂ span with sync≥thr: {stats['w_min']:.3f}–{stats['w_max']:.3f} • "
                 f"Kφ_min range on rim: {stats['k_min']:.3f}–{stats['k_max']:.3f} "
                 f"(median {stats['k_med']:.3f}, mean {stats['k_mean']:.3f}, N={stats['n']})")
        notes.insert(0, sline)
    ax3.text(0, 0.9, "Summary & Notes", fontsize=11, weight="bold")
    y = 0.7
    for line in notes:
        ax3.text(0, y, textwrap.fill(line, 110), fontsize=9)
        y -= 0.22

    pathlib.Path(OUT_PDF).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PDF)
    print(f"Saved: {OUT_PDF}")

if __name__ == "__main__":
    main()
