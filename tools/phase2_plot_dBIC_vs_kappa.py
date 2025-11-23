#!/usr/bin/env python3
import argparse, csv, matplotlib.pyplot as plt
def load(path):
    rows=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append((float(row["kappa"]), float(row["dBIC_vs_LCDM"])))
    return sorted(rows)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    for p,l in zip(args.csv, args.labels):
        xy=load(p); x=[k for k,_ in xy]; y=[v for _,v in xy]
        plt.plot(x,y,label=l)
    plt.axhline(0, linestyle="--")
    plt.xlabel("kappa"); plt.ylabel("ΔBIC vs baseline")
    plt.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=140); print("Wrote", args.out)
