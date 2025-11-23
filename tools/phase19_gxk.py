#!/usr/bin/env python3
"""
Phase-18 helper: export fitted curves for each spectrum using Phase-18 summary.

Reads:
  - outputs/phase18/lorentz_fit_summary.csv  (produced by phase18_lorentzian_fit.py)
  - spectra: e.g. outputs/phase8/spectrum_*.csv

Writes (one per OK row):
  - outputs/phase18/fitcurves/<base>_fit.csv    (omega, data, model)
  - outputs/phase18/fitcurves/<base>_fit.png    (zoomed overlay)

Notes:
  * Provide --omega-scale used during fitting, so x matches the Phase-18 run.
  * Auto-detects model per row (voigt or lorentz).
  * Optional --peak-range focuses the export window.

Safe to re-run; overwrites only its own outputs.
"""

from __future__ import annotations
import argparse, os, sys, csv, math, re, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.special import wofz

CANDIDATE_X = ["omega", "w", "freq", "x", "frequency"]
CANDIDATE_Y = ["power", "P", "amp", "amplitude", "y", "spectrum"]

def pick_xy_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    x = next((cols[c] for c in CANDIDATE_X if c in cols), None)
    y = next((cols[c] for c in CANDIDATE_Y if c in cols), None)
    if x is None or y is None:
        num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num) >= 2: x, y = num[0], num[1]
        else: raise ValueError("Could not infer (x,y) columns.")
    return x, y

def lorentz(x, x0, A, gL, b):
    gL = max(1e-12, gL)
    return b + A / (1.0 + ((x - x0)/gL)**2)

def voigt_model(x, x0, A, gL, sG, b):
    sG = max(1e-12, sG); gL = max(1e-12, gL)
    z = ((x - x0) + 1j*gL) / (sG*np.sqrt(2.0))
    V = np.real(wofz(z)) / (sG*np.sqrt(2.0*np.pi))
    return b + A * V

def sanitize_base(fname: str) -> str:
    # strip directory and common prefixes like "spectrum_"
    base = os.path.basename(fname)
    base = re.sub(r"^spectrum_","",base)
    base = re.sub(r"\.csv$","",base)
    return base

def main():
    ap = argparse.ArgumentParser(description="Export Phase-18 fitted curves per spectrum.")
    ap.add_argument("--summary", default="outputs/phase18/lorentz_fit_summary.csv",
                    help="Phase-18 summary CSV")
    ap.add_argument("--spectra-glob", default="outputs/phase8/spectrum_*.csv",
                    help="Glob for spectra used in Phase-18")
    ap.add_argument("--omega-scale", type=float, required=True,
                    help="ω scale factor used during Phase-18 fitting")
    ap.add_argument("--peak-range", type=float, nargs=2, default=None,
                    help="Optional ω window [min,max] to export/plot")
    ap.add_argument("--outdir", default="outputs/phase18/fitcurves", help="Output directory")
    ap.add_argument("--samples", type=int, default=2000, help="Resample model to N points in window")
    args = ap.parse_args()

    import glob
    spec_files = {os.path.basename(p): p for p in glob.glob(args.spectra_glob)}
    if not spec_files:
        print(f"No spectra matched: {args.spectra_glob}", file=sys.stderr); sys.exit(2)

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # read summary
    rows=[]
    with open(args.summary) as f:
        r=csv.DictReader(f)
        for d in r:
            if d.get("ok","0") == "1":
                rows.append(d)
    if not rows:
        print("No OK rows in summary; nothing to export.", file=sys.stderr); sys.exit(2)

    count=0
    for d in rows:
        fname = d["file"]
        if fname not in spec_files:
            # try with "spectrum_" prefix
            maybe = "spectrum_"+fname
            if maybe in spec_files: fname = maybe
            else:
                print(f"[WARN] Missing spectrum for {d['file']} (looked for {fname})")
                continue

        df = pd.read_csv(spec_files[fname])
        xcol, ycol = pick_xy_columns(df)
        x = df[xcol].to_numpy(dtype=float) * args.omega_scale
        y = df[ycol].to_numpy(dtype=float)

        # optional window
        if args.peak_range is not None:
            wmin, wmax = args.peak_range
            m = (x >= min(wmin,wmax)) & (x <= max(wmin,wmax))
            if m.sum() >= 10:
                x, y = x[m], y[m]

        model = d.get("model","lorentz")
        x0 = float(d["omega0"]); A=float(d["A"]); b=float(d["baseline"])
        if model == "voigt":
            gL=float(d["gamma_L_HWHM"]); sG=float(d["sigma_G"])
            yhat = voigt_model(x, x0, A, gL, sG, b)
        else:
            gL=float(d["gamma_L_HWHM"])
            yhat = lorentz(x, x0, A, gL, b)

        # export CSV (resample model on a clean grid over the x-range)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        gx = np.linspace(xmin, xmax, int(args.samples))
        if model == "voigt":
            gy = voigt_model(gx, x0, A, gL, sG, b)
        else:
            gy = lorentz(gx, x0, A, gL, b)

        base = sanitize_base(fname)
        out_csv = outdir / f"{base}_fit.csv"
        pd.DataFrame({"omega": gx, "model": gy}).to_csv(out_csv, index=False)

        # quick overlay plot (normalized for visibility)
        y0 = y - np.min(y); yh = yhat - np.min(yhat)
        if np.max(y0) > 0: y0 = y0/np.max(y0)
        if np.max(yh) > 0: yh = yh/np.max(yh)
        plt.figure(figsize=(7.5,5.0))
        plt.plot(x, y0, lw=1.0, alpha=0.7, label="data")
        plt.plot(x, yh, lw=1.4, alpha=0.95, linestyle="--", label=f"fit ({model})")
        plt.xlabel("ω")
        plt.ylabel("normalized power")
        plt.title(f"{base}: Phase-18 fit overlay")
        plt.legend()
        plt.tight_layout()
        out_png = outdir / f"{base}_fit.png"
        plt.savefig(out_png, dpi=140)
        plt.close()

        print(f"Wrote {out_csv}")
        print(f"Wrote {out_png}")
        count += 1

    print(f"Exported {count} fit curve(s). RESULT: PASS")

if __name__ == "__main__":
    main()
