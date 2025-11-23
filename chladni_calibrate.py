#!/usr/bin/env python3
import os, shutil, subprocess, csv, statistics as st
from pathlib import Path

FREQ_LIST = [1.4, 1.6, 1.8, 2.0, 2.2]
BASE = dict(N=192, steps=3600, save_every=300, freq1=1.6, switch_step=10_000_000,
            gamma=0.05, nu=0.001, A=0.25, c=1.0, alpha_k=28, drive='eigen')
SCRIPT = "chladni_cosmo_toy.py"
WORKROOT = Path("calibrate_runs")

def _flag(k: str) -> str:
    return "--" + k.replace("_", "-")

def median_spacing(out_csv: Path):
    pre = []
    with out_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            pre.append(float(row['spacing_est_dx_units']))
    return st.median(pre) if pre else float('nan')

def main():
    WORKROOT.mkdir(exist_ok=True)
    rows = []
    for f in FREQ_LIST:
        run_dir = WORKROOT / f"omega_{f:.2f}".replace('.', 'p')
        run_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy2(SCRIPT, run_dir / SCRIPT)
        args = {**BASE, "freq1": f}

        cli = ["python3", SCRIPT]
        for k, v in args.items():
            cli += [_flag(k), str(v)]

        subprocess.run(cli, cwd=run_dir, check=True)
        sp = median_spacing(run_dir / "outputs" / "dominant_spacing.csv")
        rows.append((f, sp))
        print(f"[done] Ω={f:.2f}  median spacing={sp:.4f}")

    # Fit k ≈ a*Ω + b  where k ~ 1/spacing (grid units/dx)
    import numpy as np, matplotlib.pyplot as plt
    Om = np.array([r[0] for r in rows]); spacing = np.array([r[1] for r in rows])
    k = 1.0/spacing
    a, b = np.polyfit(Om, k, 1)
    yhat = a*Om + b
    ss_res = np.sum((k - yhat)**2); ss_tot = np.sum((k - k.mean())**2)
    R2 = 1.0 - ss_res/ss_tot

    print("\n=== Calibration ===")
    print(f"k ≈ {a:.3f} * Ω + {b:.3f}    (R^2={R2:.3f})")

    fig = plt.figure(figsize=(6,4), dpi=120)
    plt.scatter(Om, k, label="runs")
    xx = np.linspace(min(Om), max(Om), 100); yy = a*xx + b
    plt.plot(xx, yy, linestyle='--', label=f"fit (R^2={R2:.2f})")
    plt.xlabel("Ω"); plt.ylabel("k ≈ 1/spacing")
    plt.title("Wavenumber vs driving frequency")
    plt.legend(); plt.tight_layout()
    plt.savefig(WORKROOT / "k_vs_Omega.png"); plt.close(fig)

if __name__ == "__main__":
    main()
