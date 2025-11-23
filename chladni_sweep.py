#!/usr/bin/env python3
import os, shutil, subprocess, csv, statistics as st
from pathlib import Path

# ---- user knobs ----
FREQ2_LIST = [1.7, 1.9, 2.1, 2.3, 2.5]  # post-switch Ω to test
BASE = dict(N=192, steps=4800, save_every=300, freq1=1.6, switch_step=2400,
            gamma=0.05, nu=0.001, A=0.25, c=1.0, alpha_k=28, drive='eigen')
SCRIPT = "chladni_cosmo_toy.py"
WORKROOT = Path("sweep_runs")  # all runs live here
# --------------------

def _flag(k: str) -> str:
    return "--" + k.replace("_", "-")

def run_one(freq2: float):
    run_dir = WORKROOT / f"freq2_{freq2:.2f}".replace('.', 'p')
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, run_dir / SCRIPT)
    args = {**BASE, "freq2": freq2}

    cli = ["python3", SCRIPT]
    for k, v in args.items():
        cli += [_flag(k), str(v)]

    # run
    subprocess.run(cli, cwd=run_dir, check=True)

    # read outputs
    out_csv = run_dir / "outputs" / "dominant_spacing.csv"
    pre, post = [], []
    with out_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            step = int(row["step"]); sp = float(row["spacing_est_dx_units"])
            (pre if step < BASE["switch_step"] else post).append(sp)

    m = lambda xs: float('nan') if not xs else st.median(xs)
    pre_m, post_m = m(pre), m(post)
    chg = (post_m - pre_m)/pre_m*100 if pre_m==pre_m and post_m==post_m else float('nan')
    return dict(freq2=freq2, pre=pre_m, post=post_m, pct_change=chg, run_dir=str(run_dir))

def main():
    WORKROOT.mkdir(exist_ok=True)
    rows = []
    for f2 in FREQ2_LIST:
        print(f"[run] freq2={f2}")
        rows.append(run_one(f2))

    print("\n=== Sweep summary ===")
    print("freq2\tpre\tpost\t%Δ")
    for r in rows:
        print(f"{r['freq2']:.3f}\t{r['pre']:.4f}\t{r['post']:.4f}\t{r['pct_change']:+.2f}")

    # save CSV + quick plot
    import matplotlib.pyplot as plt
    out_csv = WORKROOT / "sweep_results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["freq2","pre","post","pct_change","run_dir"])
        for r in rows: w.writerow([r['freq2'], r['pre'], r['post'], r['pct_change'], r['run_dir']])

    # plot post-spacing vs freq2 (expect downward slope)
    freq = [r['freq2'] for r in rows]
    post = [r['post'] for r in rows]
    fig = plt.figure(figsize=(6,4), dpi=120)
    plt.scatter(freq, post)
    import numpy as np
    x = np.array(freq); y = np.array(post)
    a, b = np.polyfit(x, y, 1)
    xx = np.linspace(min(x), max(x), 100); yy = a*xx + b
    plt.plot(xx, yy, linestyle='--')
    plt.xlabel("Ω post-switch"); plt.ylabel("Median spacing (dx units)")
    plt.title("Spacing vs frequency (mode-shift)")
    plt.tight_layout()
    figpath = WORKROOT / "sweep_plot.png"
    plt.savefig(figpath); plt.close(fig)
    print(f"\nSaved: {out_csv}\nSaved: {figpath}")

if __name__ == "__main__":
    main()
