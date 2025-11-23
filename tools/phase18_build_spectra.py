#!/usr/bin/env python3
"""
Phase-18 helper: convert residual time series -> spectrum CSVs.

Input glob (default): outputs/phase8/*residual_series.csv
Output: outputs/phase8/spectrum_<basename>.csv with columns [omega,power]

- Auto-detects numeric time + signal columns.
- Applies Hann window, removes mean, uses rFFT.
- Safe to re-run (overwrites only its own spectrum_* files).
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd

CANDS_TIME = ["t","time","x"]
CANDS_SIG  = ["resid","residual","y","signal","value","amp"]

def pick_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    t = next((cols[c] for c in CANDS_TIME if c in cols), None)
    y = next((cols[c] for c in CANDS_SIG  if c in cols), None)
    if t is None or y is None:
        # fallback: first two numeric columns
        nums = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(nums) >= 2:
            t, y = nums[0], nums[1]
        else:
            raise ValueError("Could not infer time/signal columns")
    return t, y

def make_spectrum(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 16: raise ValueError("Too few points")

    # sort by time, de-mean, Hann window
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    y = y - np.mean(y)
    w = np.hanning(len(y))
    yw = y * w

    # uniform dt estimate
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0: raise ValueError("Bad dt")

    # rFFT
    Y = np.fft.rfft(yw)
    f = np.fft.rfftfreq(len(yw), d=dt)  # cycles per unit t
    omega = 2*np.pi*f                   # angular frequency
    power = (np.abs(Y)**2) / np.sum(w*w)

    return omega, power

def main():
    ap = argparse.ArgumentParser(description="Build spectra from residual series.")
    ap.add_argument("input_glob", nargs="?", default="outputs/phase8/*residual_series.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print(f"No files matched: {args.input_glob}")
        return

    for f in files:
        try:
            df = pd.read_csv(f)
            tcol, ycol = pick_cols(df)
            om, pw = make_spectrum(df[tcol], df[ycol])
            out = os.path.join(os.path.dirname(f), "spectrum_" + os.path.basename(f).replace(".csv","") + ".csv")
            pd.DataFrame({"omega": om, "power": pw}).to_csv(out, index=False)
            print("Wrote", out)
        except Exception as e:
            print("SKIP", f, "->", e)

if __name__ == "__main__":
    main()
