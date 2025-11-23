# resonant_benchmarks.py
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

BENCH_DIR = Path.home() / "resonant-medium-project" / "data" / "benchmarks"
ABH_CSV = BENCH_DIR / "ABH_plate_benchmark.csv"
PLASMA_CSV = BENCH_DIR / "DustyPlasma_dispersion_benchmark.csv"

def fit_exp_decay(y, x):
    best = {"rss": np.inf}
    for y0 in np.linspace(min(y), max(y), 20):
        for A in np.linspace(0, max(y)-min(y), 30):
            for k in np.linspace(0.01, 1.0, 40):
                yhat = y0 + A*np.exp(-k*x)
                rss = np.sum((y - yhat)**2)
                if rss < best["rss"]:
                    best = {"y0": float(y0), "A": float(A), "k": float(k), "rss": float(rss)}
    return best

def fit_one_minus_exp(y, x, base=1.0):
    best = {"rss": np.inf}
    for A in np.linspace(0, max(y)-base, 40):
        for B in np.linspace(0.05, 5.0, 40):
            yhat = base + A*(1-np.exp(-x/B))
            rss = np.sum((y - yhat)**2)
            if rss < best["rss"]:
                best = {"base": float(base), "A": float(A), "B": float(B), "rss": float(rss)}
    return best

abh = pd.read_csv(ABH_CSV)
plasma = pd.read_csv(PLASMA_CSV)

fitE = fit_exp_decay(abh.E_anchor_ratio, abh.mode_index)
fitQ = fit_exp_decay(abh.Q, abh.mode_index)
fitL = fit_one_minus_exp(plasma.omega_L_norm, plasma.k_a)
fitT = fit_exp_decay(plasma.omega_T_norm, plasma.k_a)

out = {
    "ABH": {"E_fit": fitE, "Q_fit": fitQ},
    "DustyPlasma": {"L_fit": fitL, "T_fit": fitT}
}

out_path = BENCH_DIR / "benchmarks_fit_summary.json"
json.dump(out, open(out_path, "w"), indent=2)
print("Wrote", out_path)
