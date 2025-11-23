#!/usr/bin/env python3
"""
Phase-20: Prediction Bench for Phase-19 g×κ sim outputs

- Reads a Phase-19 CSV (defaults to outputs/phase19/gxk_sim_summary.csv)
- Checks sign/magnitude of ΔC_ell/C_ell per model kind (notch=negative, boost=positive)
- Writes:
    outputs/phase20/bench_summary.csv
    outputs/phase20/bench_report.txt
    outputs/phase20/bench_hist.png
    outputs/phase20/bench_scatter.png
- Headless-safe, re-runnable, no edits to existing code.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def safe_mkdir(p): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/phase19/gxk_sim_summary.csv",
                    help="Input CSV from Phase-19")
    ap.add_argument("--neg-range", nargs=2, type=float, default=[-0.25, -0.01],
                    help="Expected range for notch median ΔC/C")
    ap.add_argument("--pos-range", nargs=2, type=float, default=[0.01, 0.25],
                    help="Expected range for boost median ΔC/C")
    ap.add_argument("--outdir", default="outputs/phase20")
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    df = pd.read_csv(args.csv)
    if "dA_over_A" not in df.columns or "kind" not in df.columns:
        print("ERROR: input CSV missing required columns.", file=sys.stderr)
        sys.exit(2)

    # If multiple kinds present, we evaluate each kind separately.
    records = []
    report_lines = []
    for kind, sub in df.groupby("kind"):
        vals = sub["dA_over_A"].astype(float)
        med  = float(vals.median())
        mean = float(vals.mean())
        std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        frac_sign = float((vals*np.sign(med) > 0).mean()) if med != 0 else float((vals==0).mean())
        worst = sub.reindex(vals.abs().sort_values(ascending=False).index).head(3)

        if kind == "notch":
            lo, hi = args.neg_range
            expected = f"[{lo:.3f},{hi:.3f}] (negative)"
            ok = (lo <= med <= hi)
        else:  # boost
            lo, hi = args.pos_range
            expected = f"[{lo:.3f},{hi:.3f}] (positive)"
            ok = (lo <= med <= hi)

        status = "PASS" if ok else "FLAG"
        records.append(dict(kind=kind, n=len(sub), median=med, mean=mean, std=std,
                            frac_same_sign=frac_sign, status=status,
                            expected_range=expected))

        # Report chunk
        report_lines += [
            f"=== KIND: {kind} ===",
            f"n={len(sub)}  median={med:.5f}  mean={mean:.5f}  std={std:.5f}",
            f"Expected median range: {expected}",
            f"Fraction with same sign as median: {frac_sign:.3f}",
            f"Status: {status}",
            "Top-3 by |ΔC/C|:",
            worst.to_string(index=False),
            ""
        ]

        # Plots per kind
        # Histogram
        plt.figure(figsize=(7.0,4.6), dpi=140)
        plt.hist(vals, bins=40)
        plt.axvline(0, ls="--", lw=1)
        plt.xlabel("ΔC_ℓ / C_ℓ")
        plt.ylabel("Count")
        plt.title(f"Phase-20: ΔC_ℓ/C_ℓ Histogram ({kind})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"bench_hist_{kind}.png"))

        # Scatter (ω0 vs ΔC/C), color by Q if present
        plt.figure(figsize=(7.0,4.6), dpi=140)
        x = sub["omega0"] if "omega0" in sub.columns else pd.Series([0]*len(sub))
        y = sub["dA_over_A"]
        if "Q" in sub.columns:
            sc = plt.scatter(x, y, c=sub["Q"], s=22, edgecolor="none")
            cbar = plt.colorbar(sc, label="Q")
        else:
            plt.scatter(x, y, s=22, edgecolor="none")
        plt.axhline(0, ls="--", lw=1)
        plt.xlabel("ω₀")
        plt.ylabel("ΔC_ℓ / C_ℓ")
        plt.title(f"Phase-20: ΔC_ℓ/C_ℓ vs ω₀ ({kind})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"bench_scatter_{kind}.png"))

    bench_df = pd.DataFrame.from_records(records)
    bench_csv = os.path.join(args.outdir, "bench_summary.csv")
    bench_df.to_csv(bench_csv, index=False)

    # Overall status = PASS if all kinds passed
    overall = "PASS" if all(r["status"]=="PASS" for r in records) else "FLAG"
    report_path = os.path.join(args.outdir, "bench_report.txt")
    with open(report_path, "w") as f:
        f.write("Phase-20: Prediction Bench\n")
        f.write(f"Input: {args.csv}\n")
        f.write(f"Overall: {overall}\n\n")
        for line in report_lines:
            f.write(line+"\n")

    print(f"Wrote {bench_csv}")
    print(f"Wrote {report_path}")
    for kind in df["kind"].unique():
        print(f"Wrote {os.path.join(args.outdir, f'bench_hist_{kind}.png')}")
        print(f"Wrote {os.path.join(args.outdir, f'bench_scatter_{kind}.png')}")
    print(f"RESULT: {overall}")

if __name__ == "__main__":
    main()
