#!/usr/bin/env python3
"""
Phase-2: Hubble vs Resonant Suppression Test (v3, robust)

What it does
------------
- Compares a Hubble-only growth shape (LCDM-like baseline) against a resonant-suppressed
  shape with parameters (kappa, tau).
- For *each* grid point, fits a single multiplicative amplitude A* by weighted least squares,
  so comparisons focus on *shape* not absolute normalization.
- Writes CSV/TXT/JSON with chi2, AIC, BIC, ΔAIC/ΔBIC vs baseline, WWI, and VERDICT.

Why v3?
-------
- Robust CSV parsing: never overwrites parsed floats with string columns.
- Optional baseline column (e.g., 'fs8_lcdm_repaired') if present in your Phase-2 tables.
- Ignores rows with NaNs after parsing (reports how many were kept).

CLI Examples
------------
# With a baseline column present in your CSV:
PYTHONPATH=. python3 tools/phase2_hubble_suppression_test_v3.py \
  --fs8-data outputs/phase2/fs8_normalized.csv \
  --baseline-col fs8_lcdm_repaired \
  --out-prefix outputs/phase2/hubble_vs_resonant/testA_v3 \
  --kappa-range 0.00 0.25 11 \
  --tau-range   0.00 0.40  9

# Without baseline column:
PYTHONPATH=. python3 tools/phase2_hubble_suppression_test_v3.py \
  --fs8-data outputs/phase2/fs8_normalized.csv \
  --out-prefix outputs/phase2/hubble_vs_resonant/testA_v3
"""
import argparse, csv, json, math, os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


def safe_float(x, default=np.nan):
    try:
        return float(str(x).strip())
    except Exception:
        return default


def load_fs8_table(path: str,
                   ycol: str = "fs8",
                   zcol: str = "z",
                   sig_candidates: Tuple[str, ...] = ("sigma", "err")) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            z = safe_float(row.get(zcol))
            y = safe_float(row.get(ycol))
            sig = np.nan
            for name in sig_candidates:
                if name in row and row[name] not in (None, ""):
                    sig = safe_float(row[name])
                    break
            if np.isnan(sig):
                sig = 0.05  # conservative default if not supplied
            # keep extra columns but DO NOT overwrite parsed numeric keys
            extra = {k: v for k, v in row.items() if k not in (zcol, ycol) + sig_candidates}
            rows.append({"z": z, "fs8": y, "sigma": sig, **extra})
    # drop rows with NaNs
    clean = [d for d in rows if np.isfinite(d["z"]) and np.isfinite(d["fs8"]) and np.isfinite(d["sigma"]) and d["sigma"] > 0]
    dropped = len(rows) - len(clean)
    if dropped > 0:
        print(f"[hsr v3] Dropped {dropped} rows with NaNs or nonpositive sigma.")
    return clean


def lcdm_curve(z: float, Om0: float) -> float:
    # very crude stand-in (use baseline-col if you have it)
    a = 1.0 / (1.0 + z)
    Omz = Om0 / (Om0 + (1.0 - Om0) * a**3)
    return 0.5 * (Omz**0.55)


def get_baseline_series(data: List[Dict[str, Any]], Om0: float, baseline_col: Optional[str]) -> np.ndarray:
    if baseline_col and (baseline_col in data[0]):
        base = np.array([safe_float(d[baseline_col]) for d in data])
        # if any NaNs in provided baseline, fall back per-row
        bad = ~np.isfinite(base)
        if bad.any():
            base[bad] = np.array([lcdm_curve(d["z"], Om0) if b else base[i]
                                  for i, (d, b) in enumerate(zip(data, bad))])
    else:
        base = np.array([lcdm_curve(d["z"], Om0) for d in data])
    return base


def resonant_suppression_factor(z: float, kappa: float, tau: float) -> float:
    # Smooth mid-z kernel that peaks near a~0.5 (z~1) and widens with tau
    a = 1.0 / (1.0 + z)
    G = a * (1.0 - a) * (1.0 + 0.5 * tau)
    return 1.0 / (1.0 + kappa * G)


def model_shape_series(data: List[Dict[str, Any]], base_series: np.ndarray, kappa: float, tau: float) -> np.ndarray:
    fac = np.array([resonant_suppression_factor(d["z"], kappa, tau) for d in data])
    return base_series * fac


def fit_amplitude_WLS(y: np.ndarray, m: np.ndarray, s: np.ndarray) -> float:
    w = 1.0 / np.square(s)
    num = np.sum(w * y * m)
    den = np.sum(w * np.square(m))
    return (num / den) if den > 0 else 1.0


def chi2(y: np.ndarray, m: np.ndarray, s: np.ndarray) -> float:
    return float(np.sum(np.square((y - m) / s)))


def aic(c2: float, k: int) -> float:
    return c2 + 2 * k


def bic(c2: float, k: int, n: int) -> float:
    return c2 + k * math.log(n)


def wwi_from_deltas(dA: float, dB: float) -> float:
    # Your heuristic
    return 100 * min(1.0, max(0, -dA / 10.0), max(0, -dB / 10.0))


def run_suite(fs8_csv: str,
              out_prefix: str,
              kappa_grid: np.ndarray,
              tau_grid: np.ndarray,
              Om0: float,
              baseline_col: Optional[str] = None,
              save_preds: bool = False) -> Tuple[str, str, str]:
    data = load_fs8_table(fs8_csv)
    n = len(data)
    if n == 0:
        raise RuntimeError("No valid rows after cleaning; check your CSV.")

    y = np.array([d["fs8"] for d in data])
    s = np.array([d["sigma"] for d in data])
    base = get_baseline_series(data, Om0, baseline_col)

    # Baseline (Hubble-only) with amplitude fit
    m0_shape = base
    A0 = fit_amplitude_WLS(y, m0_shape, s)
    y0 = A0 * m0_shape
    chi2_0 = chi2(y, y0, s)
    k0 = 1  # amplitude only (Om0 fixed here)
    AIC0 = aic(chi2_0, k0)
    BIC0 = bic(chi2_0, k0, n)

    rows = []
    best = {"chi2": 1e99}
    best_preds = None

    for kap in kappa_grid:
        for tau in tau_grid:
            m_shape = model_shape_series(data, base, float(kap), float(tau))
            A = fit_amplitude_WLS(y, m_shape, s)
            yhat = A * m_shape
            c2 = chi2(y, yhat, s)

            # parameters counted: A + (kappa if nonzero) + (tau if nonzero)
            k = 1 + (1 if kap != 0 else 0) + (1 if tau != 0 else 0)
            AIC = aic(c2, k); BIC = bic(c2, k, n)
            dA = AIC - AIC0; dB = BIC - BIC0
            WWI = wwi_from_deltas(dA, dB)

            row = {
                "kappa": float(kap), "tau": float(tau),
                "A_amp": float(A), "chi2": float(c2),
                "AIC": float(AIC), "BIC": float(BIC),
                "dAIC_vs_LCDM": float(dA),
                "dBIC_vs_LCDM": float(dB),
                "WWI": float(WWI)
            }
            rows.append(row)
            if c2 < best["chi2"]:
                best = row
                if save_preds:
                    # store predictions for the best model so far
                    best_preds = [{"z": data[i]["z"], "fs8_obs": y[i], "sigma": s[i], "fs8_pred": yhat[i]}
                                  for i in range(n)]

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # CSV
    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Optional predictions of the BEST model
    if save_preds and best_preds is not None:
        pred_path = f"{out_prefix}.best_predictions.csv"
        with open(pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["z", "fs8_obs", "sigma", "fs8_pred"])
            w.writeheader(); w.writerows(best_preds)
        print(f"[hsr v3] Wrote {pred_path}")

    # TXT
    txt_path = f"{out_prefix}.txt"
    with open(txt_path, "w") as f:
        f.write("=== Phase-2: Hubble vs Resonant Suppression Test (v3) ===\n")
        f.write(f"fs8 file: {fs8_csv}\n")
        if baseline_col: f.write(f"baseline_col: {baseline_col}\n")
        f.write(f"n = {n}\n")
        f.write(f"Baseline (LCDM-like) A*={A0:.6f}  chi2={chi2_0:.3f}  AIC={AIC0:.3f}  BIC={BIC0:.3f}\n")
        f.write("\n--- Best Resonant (after amplitude fit) ---\n")
        for k in ["kappa","tau","A_amp","chi2","AIC","BIC","dAIC_vs_LCDM","dBIC_vs_LCDM","WWI"]:
            v = best[k]; f.write(f"{k}: {v:.6f}\n")
        verdict = "PASS" if (best["AIC"] < AIC0 and best["BIC"] <= BIC0) else "TIE/FAIL"
        f.write(f"\nVERDICT: {verdict}\n")

    # JSON
    json_path = f"{out_prefix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "fs8_csv": fs8_csv,
            "n": n,
            "baseline": {"A": A0, "chi2": chi2_0, "AIC": AIC0, "BIC": BIC0},
            "best": best,
            "baseline_col": baseline_col
        }, f, indent=2)

    # Console summary
    print(f"[hsr v3] Baseline: chi2={chi2_0:.3f}  AIC={AIC0:.3f}  BIC={BIC0:.3f} (A*={A0:.6f})")
    print(f"[hsr v3] Best: kappa={best['kappa']:.4f}  tau={best['tau']:.4f}  "
          f"chi2={best['chi2']:.3f}  dAIC={best['dAIC_vs_LCDM']:.3f}  dBIC={best['dBIC_vs_LCDM']:.3f}  WWI={best['WWI']:.1f}")
    print(f"[hsr v3] VERDICT: {'PASS' if (best['AIC'] < AIC0 and best['BIC'] <= BIC0) else 'TIE/FAIL'}")

    return csv_path, txt_path, json_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs8-data", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--Om0", type=float, default=0.30)
    ap.add_argument("--kappa-range", nargs=3, type=float, default=[0.00, 0.25, 11], help="start stop steps")
    ap.add_argument("--tau-range",   nargs=3, type=float, default=[0.00, 0.40,  9], help="start stop steps")
    ap.add_argument("--baseline-col", default=None, help="e.g., fs8_lcdm_repaired if present in your CSV")
    ap.add_argument("--save-preds", action="store_true", help="save best-model predictions CSV")
    args = ap.parse_args()

    k0, k1, kn = args.kappa_range
    t0, t1, tn = args.tau_range
    kappa_grid = np.linspace(k0, k1, int(kn))
    tau_grid   = np.linspace(t0, t1, int(tn))

    run_suite(
        fs8_csv=args.fs8_data,
        out_prefix=args.out_prefix,
        kappa_grid=kappa_grid,
        tau_grid=tau_grid,
        Om0=args.Om0,
        baseline_col=args.baseline_col,
        save_preds=args.save_preds
    )
