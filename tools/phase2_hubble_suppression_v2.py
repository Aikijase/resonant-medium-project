#!/usr/bin/env python3
"""
Phase-2: Hubble vs Resonant Suppression Test (v2, new file)
- Compares Hubble-only (LCDM-like) vs Resonant (extra suppression via kappa, optional tau)
- Fits a multiplicative amplitude A* by weighted least squares for each (kappa,tau)
- Robust CSV parsing (no string-overwrite of z/fs8/sigma)
- Optional --baseline-col to use your repaired LCDM series from the CSV

Outputs:
  <out-prefix>.csv  : grid results
  <out-prefix>.txt  : human-readable summary + VERDICT
  <out-prefix>.json : machine summary for dashboards

Decision rule:
  PASS if best resonant model has AIC < baseline AIC and BIC <= baseline BIC
"""

import argparse, csv, json, math, os
import numpy as np

# ------------------------- utils -------------------------

def safe_float(x, default=np.nan):
    try:
        if isinstance(x, str): x = x.strip()
        return float(x)
    except Exception:
        return float(default)

def load_fs8_table(path):
    """
    Expect columns:
      z, fs8, (sigma or err), [optional baseline column e.g. fs8_lcdm_repaired]
    Keeps any extra columns, but DOES NOT override parsed z/fs8/sigma.
    """
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            z    = safe_float(row.get("z"))
            fs8  = safe_float(row.get("fs8"))
            sig  = safe_float(row.get("sigma", row.get("err", 0.05)), 0.05)
            extra = {k: v for k, v in row.items() if k not in ("z", "fs8", "sigma", "err")}
            rows.append({"z": z, "fs8": fs8, "sigma": sig, **extra})
    # tiny clean: drop rows with NaNs
    rows = [d for d in rows if np.isfinite(d["z"]) and np.isfinite(d["fs8"]) and np.isfinite(d["sigma"]) and d["sigma"]>0]
    if not rows:
        raise ValueError("No valid rows parsed from fs8 CSV.")
    return rows

def lcdm_curve(z, Om0):
    # Crude placeholder if no baseline column is provided
    a = 1.0/(1.0+z)
    Omz = Om0 / (Om0 + (1.0-Om0)*a**3)
    return 0.5 * (Omz**0.55)

def get_baseline_series(data, Om0, baseline_col=None):
    if baseline_col and baseline_col in data[0]:
        base = np.array([safe_float(d.get(baseline_col)) for d in data])
        # if the column has NaNs, fall back to placeholder for those rows
        mask = ~np.isfinite(base)
        if mask.any():
            fallback = np.array([lcdm_curve(d["z"], Om0) for d in data])
            base[mask] = fallback[mask]
    else:
        base = np.array([lcdm_curve(d["z"], Om0) for d in data])
    return base

def resonant_suppression_factor(z, kappa, tau):
    """
    Minimal, smooth, mid-z suppression kernel:
      G(a; tau) = a(1-a)*(1 + 0.5*tau),   a = 1/(1+z)
      S = 1 / (1 + kappa *
