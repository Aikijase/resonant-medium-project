#!/usr/bin/env python3
"""
Phase-15: Robustness via Monte-Carlo jitter of Phase-8 deltas.

Reads:  outputs/phase8/phase8_metrics.csv  (reference)
Does:   N trials; in each trial, add Gaussian jitter to each lock's Δ
        with std = sigma_frac * |Δ| (default 0.10 = 10%) and check if
        ordering A > τ > κ holds.
Writes: outputs/phase15/report.txt
        outputs/phase15/hist_preserve.png (fraction preserved per sigma)

Usage:
  python3 tools/phase15_robustness.py [--sigma 0.10] [--trials 5000]
"""
import os, csv, argparse, random, math
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = "outputs/phase15"

def load_deltas(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    base = float(rows[0]["metric"])
    d = {}
    for r in rows:
        name = os.path.basename(r["stem"]).replace("joint_","").lower()
        if not r["metric"]: continue
        dy = float(r["metric"]) - base
        if name.startswith("tau"):   d["tau0"] = dy
        if name.startswith("kappa"): d["kappa0"] = dy
        if name.upper().startswith("A0"): d["A0"] = dy
    return d

def trial_preserves_order(deltas, sigma_frac, rng):
    # draw jittered deltas
    def jitter(x):
        sd = sigma_frac * max(abs(x), 1e-12)
        return rng.gauss(x, sd)
    A = jitter(deltas["A0"])
    T = jitter(deltas["tau0"])
    K = jitter(deltas["kappa0"])
    return (A > T > K)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.10, help="relative jitter std (e.g., 0.10 = 10%)")
    ap.add_argument("--trials", type=int, default=5000, help="number of Monte-Carlo trials")
    ap.add_argument("--csv", default="outputs/phase8/phase8_metrics.csv", help="reference metrics CSV")
    args = ap.parse_args()

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    d = load_deltas(args.csv)
    if not all(k in d for k in ("A0","tau0","kappa0")):
        raise SystemExit("Missing one or more deltas in reference CSV.")

    rng = random.Random(12345)
    kept = sum(trial_preserves_order(d, args.sigma, rng) for _ in range(args.trials))
    frac = kept / args.trials

    verdict = "PASS" if frac >= 0.95 else "FAIL"

    # sweep a few sigma values for a quick visual
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
    fracs = []
    for s in sigmas:
        rng2 = random.Random(2025)
        kept_s = sum(trial_preserves_order(d, s, rng2) for _ in range(2000))
        fracs.append(kept_s / 2000.0)

    with open(f"{OUTDIR}/report.txt","w") as f:
        f.write("Phase-15 robustness report\n")
        f.write(f"Reference CSV: {args.csv}\n")
        f.write(f"Deltas: A0={d['A0']}, tau0={d['tau0']}, kappa0={d['kappa0']}\n")
        f.write(f"Trials: {args.trials}, sigma={args.sigma}\n")
        f.write(f"Preserved ordering fraction: {frac:.4f}\n")
        f.write(f"Verdict (>=0.95): {verdict}\n")
        f.write("\nSigma sweep (2000 trials each):\n")
        for s,fr in zip(sigmas, fracs):
            f.write(f"  sigma={s:.2f} -> preserve={fr:.4f}\n")

    plt.figure(figsize=(5.0,3.4))
    plt.plot(sigmas, fracs, marker="o")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Relative jitter σ (fraction of |Δ|)")
    plt.ylabel("Ordering preserved fraction")
    plt.title("Phase-15: A>τ>κ robustness vs noise")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/hist_preserve.png", dpi=200)

    print("Wrote", f"{OUTDIR}/report.txt")
    print("Wrote", f"{OUTDIR}/hist_preserve.png")
    print("RESULT:", verdict)

if __name__ == "__main__":
    main()
