#!/usr/bin/env python3

"""
Softness Diagnostic for Resonant Fits
-------------------------------------
Reads a CSV or JSON (list of records) containing columns/keys:
    A, f, phi, gamma, chi2, ndof (optional), model (optional), label (optional)

Computes candidate softness indices:
    s1 = A / f**2
    s2 = A / (f**2 * (1 + gamma))
    s3 = (A * np.exp(-gamma)) / f**2

Outputs:
    - merged CSV with softness indices and delta metrics
    - PNG scatter plots: chi2 vs softness, delta-chi2 vs softness
    - Spearman correlations printed to console

Usage:
    python softness_diagnostic.py --in runs.json --out outputs/
    python softness_diagnostic.py --in sweep.csv  --lcdm-chi2 69.194

If you provide an LCDM baseline chi2 via --lcdm-chi2, Δχ² = χ²_res − χ²_LCDM
can be computed directly. Otherwise Δχ² is omitted.
"""
import argparse, json, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def _coerce_frame(inp_path: Path) -> pd.DataFrame:
    ext = inp_path.suffix.lower()
    if ext in (".json", ".jsonl"):
        # Support list-of-records JSON or JSONL (one record per line)
        try:
            data = json.loads(inp_path.read_text())
            if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                rows = data["results"]
            else:
                rows = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in inp_path.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(inp_path)
    return df

def compute_softness_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # fill missing columns safely
    for col in ["A", "f", "gamma"]:
        if col not in df:
            df[col] = np.nan
    # numeric coercion
    for col in ["A", "f", "gamma", "chi2", "ndof"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # indices
    f2 = df["f"]**2
    one_plus_gamma = 1.0 + df["gamma"].fillna(0.0)
    df["soft_s1_A_over_f2"] = df["A"] / f2
    df["soft_s2_A_over_f2_1pg"] = df["A"] / (f2 * one_plus_gamma)
    df["soft_s3_AexpnegG_over_f2"] = (df["A"] * np.exp(-df["gamma"].fillna(0.0))) / f2
    return df

def make_scatter(df: pd.DataFrame, xcol: str, ycol: str, out_png: Path, title: str):
    plt.figure()
    plt.scatter(df[xcol], df[ycol])
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV or JSON with runs")
    ap.add_argument("--out", dest="outdir", default="softness_diag_out", help="Output directory")
    ap.add_argument("--lcdm-chi2", dest="lcdm_chi2", type=float, default=None, help="Baseline LCDM chi2")
    args = ap.parse_args()

    inp_path = Path(args.inp)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = _coerce_frame(inp_path)
    df = compute_softness_indices(df)

    # Δχ² if available
    if args.lcdm_chi2 is not None and "chi2" in df:
        df["delta_chi2_vs_LCDM"] = df["chi2"] - args.lcdm_chi2

    # Save merged table
    out_csv = outdir / "softness_diag_merged.csv"
    df.to_csv(out_csv, index=False)

    # Spearman correlations
    soft_cols = [c for c in df.columns if c.startswith("soft_")]
    if "chi2" in df:
        for sc in soft_cols:
            ok = df[[sc, "chi2"]].dropna()
            if len(ok) >= 3:
                rho, p = spearmanr(ok[sc], ok["chi2"])
                print(f"[Spearman] chi2 vs {sc}: rho={rho:.3f}, p={p:.3g}")
    if "delta_chi2_vs_LCDM" in df:
        for sc in soft_cols:
            ok = df[[sc, "delta_chi2_vs_LCDM"]].dropna()
            if len(ok) >= 3:
                rho, p = spearmanr(ok[sc], ok["delta_chi2_vs_LCDM"])
                print(f"[Spearman] Δchi2 vs {sc}: rho={rho:.3f}, p={p:.3g}")

    # Plots
    if "chi2" in df:
        for sc in soft_cols:
            make_scatter(df, sc, "chi2", outdir / f"scatter_chi2_vs_{sc}.png",
                         f"chi2 vs {sc}")
    if "delta_chi2_vs_LCDM" in df:
        for sc in soft_cols:
            make_scatter(df, sc, "delta_chi2_vs_LCDM", outdir / f"scatter_dchi2_vs_{sc}.png",
                         f"Δchi2 (res−LCDM) vs {sc}")

    print(f"Wrote: {out_csv}")
    print(f"PNG plots in: {outdir}")

if __name__ == "__main__":
    main()
