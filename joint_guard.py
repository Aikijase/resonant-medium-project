#!/usr/bin/env python3
"""
joint_guard.py  —  minimal, standalone
- Loads benchmark CSVs and the fit summary JSON
- Defines eps_from_ABH and gamma_from_Q
- Renders quick overlays to outputs/
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- Paths ----------------
ROOT = Path.home() / "resonant-medium-project"
BENCH_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "outputs"

ABH_CSV = BENCH_DIR / "ABH_plate_benchmark.csv"
PLASMA_CSV = BENCH_DIR / "DustyPlasma_dispersion_benchmark.csv"
BENCH_JSON = BENCH_DIR / "benchmarks_fit_summary.json"


# ---------------- Loaders ----------------
def load_bench_csvs():
    """Return (abh_df, plasma_df); prints heads for sanity."""
    if not ABH_CSV.exists() or not PLASMA_CSV.exists():
        raise FileNotFoundError(
            f"Expected CSVs at:\n  {ABH_CSV}\n  {PLASMA_CSV}\n"
            "Create/restore them under data/benchmarks/ and re-run."
        )
    abh = pd.read_csv(ABH_CSV)
    plasma = pd.read_csv(PLASMA_CSV)
    # quick peek
    print(abh.head())
    print(plasma.head())
    return abh, plasma


def load_benchmarks_json():
    """Load fit parameters from benchmarks_fit_summary.json if present."""
    try:
        J = json.load(open(BENCH_JSON))
        print("[BENCH] Loaded benchmark fits from data/benchmarks/benchmarks_fit_summary.json")
        return J
    except FileNotFoundError:
        print("[BENCH] WARNING: benchmarks_fit_summary.json not found — skipping benchmark overlay.")
        return None


# ---------------- Helper mappings ----------------
def eps_from_ABH(mode_index: float, E_fit: dict) -> float:
    """Normalize ABH energy localization to [0,1] as ε proxy."""
    E = E_fit["y0"] + E_fit["A"] * np.exp(-E_fit["k"] * mode_index)
    Emin, Emax = E_fit["y0"], E_fit["y0"] + E_fit["A"]
    return float(np.clip((E - Emin) / max(1e-9, Emax - Emin), 0.0, 1.0))


def gamma_from_Q(mode_index: float, Q_fit: dict) -> float:
    """Map ABH Q to a damping-like γ via γ ≈ 1/Q."""
    Q = Q_fit["y0"] + Q_fit["A"] * np.exp(-Q_fit["k"] * mode_index)
    return float(max(0.0, 1.0 / max(1e-9, Q)))


# ---------------- Plots ----------------
def make_overlays(bench: dict):
    """Render two overlay figures from the benchmark fits."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ε and γ vs mode index
    E_fit = bench["ABH"]["E_fit"]
    Q_fit = bench["ABH"]["Q_fit"]
    mgrid = np.arange(1, 21)
    eps_grid = [eps_from_ABH(m, E_fit) for m in mgrid]
    gam_grid = [gamma_from_Q(m, Q_fit) for m in mgrid]

    plt.figure()
    plt.plot(mgrid, eps_grid, 'o-', label='ε_from_ABH (lab)')
    plt.plot(mgrid, gam_grid, 's--', label='γ_from_ABH ~ 1/Q (lab)')
    plt.xlabel("Mode index (lab proxy)")
    plt.ylabel("Dimensionless value")
    plt.title("Benchmark-derived ε and γ (lab analogues)")
    plt.legend()
    plt.tight_layout()
    p1 = OUT_DIR / "bench_overlay_eps_gamma.png"
    plt.savefig(p1.as_posix(), dpi=150, bbox_inches="tight")
    plt.close()

    # Dusty plasma dispersion overlay (lab-only)
    L_fit = bench["DustyPlasma"]["L_fit"]
    T_fit = bench["DustyPlasma"]["T_fit"]
    kgrid = np.linspace(0.1, 2.0, 40)
    omega_L = L_fit["base"] + L_fit["A"] * (1.0 - np.exp(-kgrid / L_fit["B"]))
    omega_T = T_fit["y0"] + T_fit["A"] * np.exp(-T_fit["k"] * kgrid)

    plt.figure()
    plt.plot(kgrid, omega_L, '-', label='ω_L / ω₀ (lab fit)')
    plt.plot(kgrid, omega_T, '--', label='ω_T / ω₀ (lab fit)')
    plt.xlabel("k·a (lab)")
    plt.ylabel("ω / ω₀ (lab)")
    plt.title("Dusty Plasma Dispersion (lab analogue)")
    plt.legend()
    plt.tight_layout()
    p2 = OUT_DIR / "bench_overlay_plasma_dispersion.png"
    plt.savefig(p2.as_posix(), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wrote {p1.as_posix()}")
    print(f"Wrote {p2.as_posix()}")
#     print(f"chi2={chi2} A={best_params.get('A')} f={best_params.get('f')} phi={best_params.get('phi')} gamma={best_params.get('gamma')}")


# ---------------- Main ----------------
def main():
    # load the raw CSVs (sanity print)
    _abh, _plasma = load_bench_csvs()

    # load fit params (optional)
    bench = load_benchmarks_json()

    # generate overlays if fits available
    if bench is not None:
        make_overlays(bench)


if __name__ == "__main__":
    main()
