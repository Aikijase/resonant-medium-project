#!/usr/bin/env python3
import csv, pathlib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_points(path):
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        pts=[]
        for row in r:
            try:
                pts.append({
                    "target": float(row["target"]),
                    "omega2": float(row["omega2"]),
                    "Kphi_mid": float(row["Kphi_mid"]),
                    "dK_domega2": float(row["dK_domega2"]) if row["dK_domega2"] not in ("", "nan") else np.nan
                })
            except Exception:
                pass
        return pts

def main():
    import argparse
    ap=argparse.ArgumentParser(description="Phase-23 plots")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--outdir", default="outputs/phase23")
    ap.add_argument("--prefix", default="p23")
    args=ap.parse_args()

    pts=read_points(args.drift_csv)
    if not pts:
        raise SystemExit(f"No points in {args.drift_csv}")

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    by_t=defaultdict(list)
    for p in pts:
        by_t[p["target"]].append(p)
    for t in by_t:
        by_t[t].sort(key=lambda x:x["omega2"])

    # 1) Kmid vs w2 per target
    fig1=plt.figure()
    for t, lst in sorted(by_t.items()):
        w2=[p["omega2"] for p in lst]
        km=[p["Kphi_mid"] for p in lst]
        plt.plot(w2, km, marker="o", label=f"target={t:g}")
    plt.xlabel(r"$\omega^2$")
    plt.ylabel(r"$K_{\phi,\mathrm{mid}}$")
    plt.title("Midline Kφ vs ω²")
    plt.legend()
    kplot = outdir/(args.prefix+"_kmid_vs_w2.png")
    fig1.savefig(kplot, dpi=160, bbox_inches="tight")
    plt.close(fig1)

    # 2) dK/dw2 per target
    fig2=plt.figure()
    for t, lst in sorted(by_t.items()):
        w2=[p["omega2"] for p in lst]
        dk=[p["dK_domega2"] for p in lst]
        plt.plot(w2, dk, marker="o", label=f"target={t:g}")
    plt.xlabel(r"$\omega^2$")
    plt.ylabel(r"$dK_\phi/d\omega^2$")
    plt.title("Drift dKφ/dω² per target")
    plt.legend()
    dplot = outdir/(args.prefix+"_dkdw2_vs_w2.png")
    fig2.savefig(dplot, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    # 3) heatmap |dK/dw2|
    targets=sorted(by_t.keys())
    w2s=sorted({p["omega2"] for p in pts})
    M=np.full((len(targets), len(w2s)), np.nan)
    ti={t:i for i,t in enumerate(targets)}
    wi={w:i for i,w in enumerate(w2s)}
    for p in pts:
        M[ti[p["target"]], wi[p["omega2"]]]=abs(p["dK_domega2"])
    fig3=plt.figure()
    plt.imshow(M, aspect="auto", origin="lower",
               extent=[min(w2s), max(w2s), min(targets), max(targets)])
    plt.colorbar(label=r"$|dK_\phi/d\omega^2|$")
    plt.xlabel(r"$\omega^2$")
    plt.ylabel("target")
    plt.title("Drift magnitude heatmap")
    hplot = outdir/(args.prefix+"_drift_heatmap.png")
    fig3.savefig(hplot, dpi=160, bbox_inches="tight")
    plt.close(fig3)

    print({"kmid_plot": str(kplot), "dkdw2_plot": str(dplot), "heatmap": str(hplot)})

if __name__=="__main__":
    main()
