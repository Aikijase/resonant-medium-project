#!/usr/bin/env python3
"""
Phase 9 — Minimal PDF Report (Lock → Verify → Style)
Collects overlay plots, beta frontier, FFT overlay (if present), and a text page
summarising the report_card + dominant frequencies into a single PDF.

Usage:
  env PYTHONPATH=. python3 tools/phase9_make_pdf.py \
    --report-card outputs/phase9/report_card.txt \
    --overlay-ts outputs/phase9/overlay_timeseries.png \
    --overlay-phase outputs/phase9/overlay_phase.png \
    --beta-frontier outputs/phase9/beta_sweep_frontier.png \
    --fft-overlay outputs/phase9/fft_overlay.png \
    --fft-dominant outputs/phase9/fft_dominant.csv \
    --out outputs/phase9/Phase9_Report.pdf
"""
import argparse, os, textwrap, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def add_text_page(pdf, title, body):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.08, 0.92, title, fontsize=16, weight="bold")
    y = 0.86
    for para in body:
        wrapped = textwrap.fill(para, 110)
        fig.text(0.08, y, wrapped, fontsize=10, va="top")
        y -= 0.05 + 0.015*(len(wrapped)//110)
    pdf.savefig(fig); plt.close(fig)

def add_image_page(pdf, title, path):
    if not path or not os.path.exists(path):
        return
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=14, weight="bold")
    img = plt.imread(path)
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.82])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig); plt.close(fig)

def read_report_card(path):
    if not path or not os.path.exists(path):
        return []
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().strip().splitlines()
    # Return last block (the table)
    return lines[-5:] if len(lines) >= 5 else lines

def read_fft_dom(path):
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df[["run","dom_freq","dom_mag"]]
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-card", required=False)
    ap.add_argument("--overlay-ts", required=False)
    ap.add_argument("--overlay-phase", required=False)
    ap.add_argument("--beta-frontier", required=False)
    ap.add_argument("--fft-overlay", required=False)
    ap.add_argument("--fft-dominant", required=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with PdfPages(args.out) as pdf:
        # Cover
        add_text_page(pdf, "Phase 9 — Unified Resonant Operator",
            ["Lock report for core artifacts and metrics.",
             "Physical loop: amplitude-rich near resonance; Psychological loop: over-damped stability; Learning loop: adaptive smoothing.",
             "This PDF collates overlays, the β frontier, and spectral fingerprints."])

        # Report card text page
        rc_lines = read_report_card(args.report_card)
        if rc_lines:
            add_text_page(pdf, "Report Card (verbatim excerpt)", rc_lines)

        # Key images
        add_image_page(pdf, "Overlay — Time Series", args.overlay_ts)
        add_image_page(pdf, "Overlay — Phase Space", args.overlay_phase)
        add_image_page(pdf, "Amplitude–Error Frontier vs β", args.beta_frontier)

        # FFT summary
        add_image_page(pdf, "FFT — Overlay", args.fft_overlay)
        fft_tab = read_fft_dom(args.fft_dominant)
        if fft_tab is not None:
            body = ["Dominant frequency per run (cycles/sample):"]
            for _,r in fft_tab.iterrows():
                body.append(f"  • {r['run']}: f*={r['dom_freq']:.4g}  |FFT|={r['dom_mag']:.4g}")
            add_text_page(pdf, "FFT — Dominant Lines", body)

    print(f"[phase9] PDF saved: {args.out}")

if __name__ == "__main__":
    main()
