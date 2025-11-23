#!/usr/bin/env python3
import json, argparse, pathlib
import matplotlib.pyplot as plt
import csv

def load_bands(path):
    with open(path) as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands-json", required=True)
    ap.add_argument("--uncertainty-csv", required=True)
    ap.add_argument("--outdir", default="outputs/phase22")
    ap.add_argument("--prefix", default="p22")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    bands = load_bands(args.bands_json)

    # Figure 1: bands with CI ribbon
    plt.figure()
    for b in bands:
        w2 = b["omega2"]
        mid = b["Kphi_mid"]
        lo  = b["Kphi_CI_lo"]
        hi  = b["Kphi_CI_hi"]
        t = b["target"]
        plt.plot(w2, mid, label=f"target {t:.2f}")
        plt.fill_between(w2, lo, hi, alpha=0.2)
    plt.xlabel(r"$\omega^2$")
    plt.ylabel(r"$K_\phi$")
    plt.title("Phase-22: Uncertainty Bands (95% CI)")
    plt.legend()
    fig1 = outdir / f"{args.prefix}_bands_CI.png"
    plt.savefig(fig1, dpi=160, bbox_inches="tight")

    # Figure 2: σ_Kphi vs ω^2 per target (scatter)
    data = {}
    with open(args.uncertainty_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["target"])
            data.setdefault(t, {"w2":[], "sig":[]})
            data[t]["w2"].append(float(row["omega2"]))
            data[t]["sig"].append(float(row["sigma_Kphi"]))
    plt.figure()
    for t, d in sorted(data.items()):
        plt.plot(d["w2"], d["sig"], marker="o", linestyle="-", label=f"target {t:.2f}")
    plt.xlabel(r"$\omega^2$")
    plt.ylabel(r"$\sigma_{K_\phi}$")
    plt.title("Local uncertainty in $K_\\phi$ along bands")
    plt.legend()
    fig2 = outdir / f"{args.prefix}_sigma_map.png"
    plt.savefig(fig2, dpi=160, bbox_inches="tight")

    print(json.dumps({
        "bands_plot": str(fig1),
        "sigma_plot": str(fig2)
    }, indent=2))

if __name__ == "__main__":
    main()
