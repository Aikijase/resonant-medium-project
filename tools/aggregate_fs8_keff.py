#!/usr/bin/env python3
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="outputs/fs8_keff_*_results.csv",
                    help="Glob for corrected results (default: outputs/fs8_keff_*_results.csv)")
    ap.add_argument("--baseline", default="outputs/fs8_visc_baseline_results.csv",
                    help="Baseline results CSV (no viscous correction)")
    ap.add_argument("--out-root", default="outputs/fs8_keff_compare",
                    help="Output root (no extension)")
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)[["z","fs8_th"]].rename(columns={"fs8_th":"fs8_base"})
    rows = []

    csvs = sorted(glob.glob(args.pattern))
    series = {"z": base["z"], "fs8_base": base["fs8_base"]}
    labels = []
    for path in csvs:
        # expect name like outputs/fs8_keff_0_20_results.csv
        tag = os.path.basename(path).replace("fs8_keff_","").replace("_results.csv","").replace("_",".")
        df = pd.read_csv(path)[["z","fs8_th"]].rename(columns={"fs8_th":f"fs8_keff_{tag}"})
        series[f"fs8_keff_{tag}"] = pd.merge(base[["z"]], df, on="z", how="left")[f"fs8_keff_{tag}"]
        labels.append(tag)

    out = pd.DataFrame(series).sort_values("z").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_root), exist_ok=True)
    out_csv = f"{args.out_root}.csv"
    out.to_csv(out_csv, index=False)

    # Plot ratios vs z
    plt.figure()
    for tag in labels:
        col = f"fs8_keff_{tag}"
        ratio = out[col] / out["fs8_base"]
        plt.plot(out["z"], ratio, label=f"k_eff={tag} h/Mpc")
    plt.axhline(1.0, ls="--", lw=1)
    plt.xlabel("z"); plt.ylabel("fσ8 (visc) / fσ8 (baseline)")
    plt.title("Impact of viscous DM vs baseline")
    plt.legend()
    fig = f"{args.out_root}.png"
    plt.tight_layout(); plt.savefig(fig, dpi=140); plt.close()

    print("Wrote:")
    print(" ", out_csv)
    print(" ", fig)

if __name__ == "__main__":
    main()
