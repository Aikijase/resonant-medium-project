#!/usr/bin/env python3
import argparse, pathlib, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description="Phase-26 plotting")
    ap.add_argument("--worlds-csv", default="outputs/phase26/p26_worlds.csv")
    ap.add_argument("--outdir", default="outputs/phase26")
    ap.add_argument("--prefix", default="p26")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.worlds_csv)

    # no-slip hist by target
    for tgt, g in df.groupby("target"):
        plt.figure()
        g["ok"]=(g["slipped"]=="no").astype(int)
        g["ok"].hist(bins=2)
        plt.title(f"Target {tgt:.2f}: no-slip vs slip counts")
        plt.xlabel("no-slip=1 / slip=0"); plt.ylabel("count")
        p=outdir/(args.prefix+f"_t{tgt:.2f}_noslip_hist.png")
        plt.savefig(p, dpi=160, bbox_inches="tight"); plt.close()

    # time-to-slip distribution
    df2=df.copy()
    df2["t_slip"]=pd.to_numeric(df2["t_slip"], errors="coerce")
    plt.figure()
    df2["t_slip"].dropna().hist(bins=20)
    plt.xlabel("t_slip"); plt.ylabel("count"); plt.title("Time-to-slip distribution")
    p2=outdir/(args.prefix+"_t_slip_hist.png")
    plt.savefig(p2, dpi=160, bbox_inches="tight"); plt.close()

if __name__=="__main__":
    main()
