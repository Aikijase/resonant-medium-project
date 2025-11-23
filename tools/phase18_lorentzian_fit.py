#!/usr/bin/env python3
"""
Phase-18: Lorentzian/Voigt fit of spectral peak(s) from Phase-8.

- Batch-process spectra (default: outputs/phase8/*.csv)
- Auto-detect dominant peak per file
- Fit:
    (a) Lorentzian + constant baseline  [default]
    (b) Voigt (Lorentz⊗Gaussian) + constant baseline  [--voigt]
- Writes:
    outputs/phase18/lorentz_fit_summary.csv
    outputs/phase18/lorentz_fit_report.txt
    outputs/phase18/lorentz_fit_plot.png
- Safe to re-run; never edits existing files.
"""

from __future__ import annotations
import argparse, csv, glob, os, sys
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz
import matplotlib.pyplot as plt


# ---------- Models ----------
# Lorentzian with baseline: y = b + A / (1 + ((x - x0)/gamma_L)^2)
def lorentz_with_baseline(x, x0, A, gamma_L, b):
    gamma_L = max(1e-12, gamma_L)
    return b + A / (1.0 + ((x - x0)/gamma_L)**2)

# Voigt (normalized) with baseline:
# V(x) = Re[wofz(z)] / (sigma_G*sqrt(2*pi)),  z = ((x-x0)+i*gamma_L)/(sigma_G*sqrt(2))
# Model: y = b + A * V(x)
def voigt_with_baseline(x, x0, A, gamma_L, sigma_G, b):
    sigma_G = max(1e-12, sigma_G)
    gamma_L = max(1e-12, gamma_L)
    z = ((x - x0) + 1j*gamma_L) / (sigma_G*np.sqrt(2.0))
    V = np.real(wofz(z)) / (sigma_G*np.sqrt(2.0*np.pi))
    return b + A * V


# ---------- Utils ----------
CANDIDATE_X = ["omega", "w", "freq", "x", "frequency"]
CANDIDATE_Y = ["power", "P", "amp", "amplitude", "y", "spectrum"]

def pick_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    x = next((cols[c] for c in CANDIDATE_X if c in cols), None)
    y = next((cols[c] for c in CANDIDATE_Y if c in cols), None)
    if x is None or y is None:
        # Fallback: first two numeric columns
        num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num) >= 2:
            x, y = num[0], num[1]
        else:
            raise ValueError("Could not infer (x,y) columns.")
    return x, y

def moving_average(y: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1: return y
    win = min(win, max(1, len(y)//20))
    if win <= 1: return y
    k = np.ones(win)/win
    return np.convolve(y, k, mode="same")

def halfmax_width(x: np.ndarray, y: np.ndarray, x0: float, y0: float, baseline: float) -> float:
    hm = baseline + (y0 - baseline)/2.0
    left = x[0]
    for i in range(len(x)-1, -1, -1):
        if x[i] <= x0 and y[i] <= hm:
            left = x[i]; break
    right = x[-1]
    for i in range(len(x)):
        if x[i] >= x0 and y[i] <= hm:
            right = x[i]; break
    return max(1e-6, right - left)

@dataclass
class FitResult:
    file: str
    ok: bool
    x0: float = np.nan
    A: float = np.nan
    gamma_L: float = np.nan   # Lorentz HWHM
    sigma_G: float = np.nan   # Gaussian sigma
    baseline: float = np.nan
    FWHM: float = np.nan
    Q: float = np.nan
    chi2: float = np.nan
    ndof: int = 0
    rchi2: float = np.nan
    note: str = ""
    model: str = "lorentz"

def guess_initials(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    y_s = moving_average(y, 7)
    b0 = float(np.median(y_s))
    thr = b0 + np.std(y_s)
    peaks, _ = find_peaks(y_s, height=thr, distance=max(5, len(x)//50))
    if len(peaks) == 0:
        imax = int(np.argmax(y_s))
    else:
        imax = int(peaks[np.argmax(y_s[peaks])])
    x0 = float(x[imax])
    y0 = float(y_s[imax])
    A0 = max(1e-9, y0 - b0)
    fwhm0 = halfmax_width(x, y_s, x0, y0, b0)
    gamma0 = max(1e-6, 0.5 * fwhm0)  # Lorentz HWHM guess
    return x0, A0, gamma0, b0


# Voigt FWHM approximation (Olivero & Longbothum):
# FWHM_V ≈ 0.5346*ΓL + sqrt(0.2166*ΓL^2 + ΓG^2)
# where ΓL = 2*gamma_L, ΓG = 2*sqrt(2*ln2)*sigma_G
SQRT4LN2 = np.sqrt(4.0*np.log(2.0))
def voigt_fwhm(gamma_L: float, sigma_G: float) -> float:
    Gamma_L = 2.0*max(1e-12, gamma_L)
    Gamma_G = 2.0*SQRT4LN2*max(1e-12, sigma_G)
    return 0.5346*Gamma_L + np.sqrt(0.2166*Gamma_L*Gamma_L + Gamma_G*Gamma_G)


def fit_one_spectrum(x: np.ndarray, y: np.ndarray, fname: str, use_voigt: bool) -> FitResult:
    fr = FitResult(file=os.path.basename(fname), ok=False, model=("voigt" if use_voigt else "lorentz"))
    try:
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 10:
            fr.note = "Too few points"
            return fr

        x0, A0, g0, b0 = guess_initials(x, y)

        if use_voigt:
            # Initial Gaussian width ~ Lorentz for stability
            sg0 = max(1e-6, g0)
            bounds = ([float(np.min(x)), 0.0, 1e-9, 1e-9, -np.inf],
                      [float(np.max(x)), np.inf, (float(np.max(x))-float(np.min(x))), (float(np.max(x))-float(np.min(x))), np.inf])
            p0 = [x0, A0, g0, sg0, b0]
            func = voigt_with_baseline
        else:
            bounds = ([float(np.min(x)), 0.0, 1e-9, -np.inf],
                      [float(np.max(x)), np.inf, (float(np.max(x))-float(np.min(x))), np.inf])
            p0 = [x0, A0, g0, b0]
            func = lorentz_with_baseline

        popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=30000)

        # Predictions & robust chi²
        yhat = func(x, *popt)
        resid = y - yhat
        sigma = np.median(np.abs(resid - np.median(resid))) * 1.4826
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(1e-9, np.std(resid))
        w = 1.0 / max(1e-12, sigma**2)
        chi2 = float(np.sum((resid**2) * w))
        ndof = max(1, len(x) - len(popt))
        rchi2 = chi2 / ndof

        fr.ok = True
        fr.x0 = float(popt[0]); fr.A = float(popt[1])
        if use_voigt:
            fr.gamma_L = float(popt[2]); fr.sigma_G = float(popt[3]); fr.baseline = float(popt[4])
            fr.FWHM = voigt_fwhm(fr.gamma_L, fr.sigma_G)
        else:
            fr.gamma_L = float(popt[2]); fr.baseline = float(popt[3]); fr.sigma_G = np.nan
            fr.FWHM = 2.0 * fr.gamma_L

        fr.Q = fr.x0 / max(1e-12, fr.FWHM)
        fr.chi2, fr.ndof, fr.rchi2 = chi2, ndof, rchi2
        fr.note = "OK"
        return fr

    except Exception as e:
        # If Voigt fails, try Lorentz fallback automatically once
        if use_voigt:
            try:
                fr.note = f"VOIGT_FAIL({e}); falling back to Lorentz"
                fr.model = "lorentz"
                return fit_one_spectrum(x, y, fname, use_voigt=False)
            except Exception as e2:
                fr.note = f"FAIL: {e}; fallback fail: {e2}"
                return fr
        fr.note = f"FAIL: {e}"
        return fr


def main():
    ap = argparse.ArgumentParser(description="Phase-18 Lorentzian/Voigt fit (batch).")
    ap.add_argument("input_glob", nargs="?", default="outputs/phase8/*.csv",
                    help="Glob for input spectra CSVs (default: outputs/phase8/*.csv)")
    ap.add_argument("--outdir", default="outputs/phase18", help="Output directory")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = no limit)")
    ap.add_argument("--plot-limit", type=int, default=6, help="Max series to overlay on the summary plot")
    ap.add_argument("--omega-scale", type=float, default=1.0,
                    help="Multiply all ω by this factor before fitting (default 1.0).")
    ap.add_argument("--peak-range", type=float, nargs=2, default=None,
                    help="Restrict peak search and fit to ω in [min, max]. Optional.")
    ap.add_argument("--voigt", action="store_true", help="Use Voigt (Lorentz⊗Gaussian) model with baseline.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(args.input_glob))
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        print(f"[Phase-18] No files matched: {args.input_glob}", file=sys.stderr)
        sys.exit(2)

    results: List[FitResult] = []
    plot_series = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            results.append(FitResult(file=os.path.basename(f), ok=False, note=f"READ_FAIL: {e}"))
            continue

        try:
            xcol, ycol = pick_xy_columns(df)
            x = df[xcol].to_numpy(dtype=float) * float(args.omega_scale)
            y = df[ycol].to_numpy(dtype=float)
            if args.peak_range is not None:
                wmin, wmax = args.peak_range
                m = (x >= min(wmin, wmax)) & (x <= max(wmin, wmax))
                if m.sum() >= 10:
                    x, y = x[m], y[m]
        except Exception as e:
            results.append(FitResult(file=os.path.basename(f), ok=False, note=f"COLS_FAIL: {e}"))
            continue

        fr = fit_one_spectrum(x, y, f, use_voigt=args.voigt)
        results.append(fr)

        if fr.ok and len(plot_series) < args.plot_limit:
            if args.voigt and fr.model == "voigt":
                yhat = voigt_with_baseline(x, fr.x0, fr.A, fr.gamma_L, fr.sigma_G, fr.baseline)
            else:
                yhat = lorentz_with_baseline(x, fr.x0, fr.A, fr.gamma_L, fr.baseline)
            plot_series.append((os.path.basename(f), x, y, yhat, fr.model))

    # ---- Write summary CSV ----
    sum_csv = os.path.join(args.outdir, "lorentz_fit_summary.csv")
    with open(sum_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "ok", "model", "omega0", "A", "gamma_L_HWHM", "sigma_G", "baseline",
                    "FWHM", "Q", "chi2", "ndof", "rchi2", "note"])
        for r in results:
            w.writerow([r.file, int(r.ok), r.model, r.x0, r.A, r.gamma_L, r.sigma_G, r.baseline,
                        r.FWHM, r.Q, r.chi2, r.ndof, r.rchi2, r.note])

    # ---- Write report TXT ----
    rep = os.path.join(args.outdir, "lorentz_fit_report.txt")
    ok_res = [r for r in results if r.ok]
    lines = []
    lines.append("=== Phase-18: Peak Fit Report (Lorentz/Voigt) ===")
    lines.append(f"Inputs matched: {len(files)}  |  Fits OK: {len(ok_res)}  |  Outdir: {args.outdir} | Voigt={int(args.voigt)}")
    if ok_res:
        def stat(name, arr):
            arr = np.array([v for v in arr if np.isfinite(v)])
            if len(arr)==0: return f"{name}: n=0"
            return f"{name}: mean={np.mean(arr):.6g}  std={np.std(arr):.6g}  min={np.min(arr):.6g}  max={np.max(arr):.6g}  n={len(arr)}"
        lines.append(stat("omega0", [r.x0 for r in ok_res]))
        lines.append(stat("FWHM",   [r.FWHM for r in ok_res]))
        lines.append(stat("Q",      [r.Q for r in ok_res]))
        lines.append(stat("rchi2",  [r.rchi2 for r in ok_res]))
        lines.append("")
        lines.append("Top 5 sharpest (highest Q):")
        topQ = sorted(ok_res, key=lambda r: (-r.Q, r.rchi2))[:5]
        for r in topQ:
            lines.append(f"  {r.file:40s}  [{r.model}]  ω0={r.x0:.6g}  FWHM={r.FWHM:.6g}  Q={r.Q:.6g}  rχ²={r.rchi2:.3f}")
    else:
        lines.append("No successful fits.")

    lines.append("\nPer-file summary:")
    for r in results:
        if r.ok:
            lines.append(f"- {r.file}: [{r.model}] ω0={r.x0:.6g}, FWHM={r.FWHM:.6g}, Q={r.Q:.6g}, rχ²={r.rchi2:.3f}")
        else:
            lines.append(f"- {r.file}: FIT_FAIL ({r.note})")

    with open(rep, "w") as fh:
        fh.write("\n".join(lines))

    # ---- Plot overlay ----
    png = os.path.join(args.outdir, "lorentz_fit_plot.png")
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    for label, x, y, yhat, model in plot_series:
        y0 = y - np.min(y); yh0 = yhat - np.min(yhat)
        if np.max(y0) > 0:  y0  = y0  / np.max(y0)
        if np.max(yh0) > 0: yh0 = yh0 / np.max(yh0)
        ax.plot(x, y0, lw=1.0, alpha=0.65, label=f"data: {label}")
        ax.plot(x, yh0, lw=1.2, alpha=0.95, linestyle="--", label=f"{model}: {label}")
    ax.set_xlabel("ω (frequency)")
    ax.set_ylabel("normalized power (arb.)")
    ax.set_title("Phase-18 peak fits (overlay; normalized)")
    if plot_series:
        ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()

    print(f"Wrote {sum_csv}")
    print(f"Wrote {rep}")
    print(f"Wrote {png}")
    print("RESULT: PASS" if ok_res else "RESULT: FAIL (no successful fits)")


if __name__ == "__main__":
    main()
