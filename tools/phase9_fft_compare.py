#!/usr/bin/env python3
"""
Phase 9 — FFT Compare
Reads each CSV, extracts the primary numeric signal (first numeric column),
computes |FFT|, saves per-run spectrum PNGs + a combined overlay, and writes a
small table of dominant frequencies.

Usage:
  env PYTHONPATH=. python3 tools/phase9_fft_compare.py \
    outputs/phase9/physical_run.csv \
    outputs/phase9/psychological_run_noscipy.csv \
    outputs/phase9/learning_run_noscipy.csv \
    --out-prefix outputs/phase9/fft
"""
import argparse, os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

def read_signal(path):
    df = pd.read_csv(path)
    num = df.select_dtypes("number")
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns in {path}")
    s = num.iloc[:,0].to_numpy(dtype=float)
    s = s[~np.isnan(s)]
    return s

def fft_mag(signal):
    N = len(signal)
    if N < 8:
        return np.array([]), np.array([])
    # assume unit sample spacing (Δt=1 arbitrary); frequencies in cycles/sample
    S = np.fft.rfft(signal - np.mean(signal))
    f = np.fft.rfftfreq(N, d=1.0)
    mag = np.abs(S)
    return f, mag

def save_single_spectrum(f, mag, title, out_png):
    plt.figure(figsize=(7,5))
    plt.plot(f, mag)
    plt.xlabel("Frequency (cycles/sample)")
    plt.ylabel("|FFT|")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    dom_rows = []
    overlay = []

    for p in args.inputs:
        name = os.path.splitext(os.path.basename(p))[0]
        s = read_signal(p)
        f, mag = fft_mag(s)
        if f.size == 0:
            continue
        # ignore DC (index 0)
        idx = int(np.argmax(mag[1:]) + 1) if mag.size > 1 else 0
        dom_f = float(f[idx]) if f.size else None
        dom_mag = float(mag[idx]) if mag.size else None
        dom_rows.append(dict(run=name, N=len(s), dom_freq=dom_f, dom_mag=dom_mag))

        out_png = f"{args.out_prefix}_{name}.png"
        save_single_spectrum(f, mag, f"Phase 9 — FFT: {name}", out_png)
        overlay.append((name, f, mag))

    # overlay
    if overlay:
        plt.figure(figsize=(7,5))
        for name,f,mag in overlay:
            plt.plot(f, mag, label=name)
        plt.xlabel("Frequency (cycles/sample)")
        plt.ylabel("|FFT|")
        plt.title("Phase 9 — FFT Overlay")
        plt.legend(); plt.grid(True, alpha=0.4)
        plt.tight_layout()
        overlay_png = f"{args.out_prefix}_overlay.png"
        plt.savefig(overlay_png, dpi=200)
        plt.close()
        print(f"[phase9] Saved: {overlay_png}")

    # summary CSV/JSON
    tab_csv = f"{args.out_prefix}_dominant.csv"
    pd.DataFrame(dom_rows).to_csv(tab_csv, index=False)
    mani_json = f"{args.out_prefix}_manifest.json"
    json.dump({"inputs":[os.path.basename(p) for p in args.inputs],
               "dominant_table": os.path.basename(tab_csv)}, open(mani_json,"w"), indent=2)
    print(f"[phase9] Saved: {tab_csv}")
    print(f"[phase9] Saved: {mani_json}")

if __name__ == "__main__":
    main()
