#!/usr/bin/env python3
import argparse, pathlib, pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser(description="Phase-28 plotting")
    ap.add_argument("--endurance-csv", default="outputs/phase28/p28_endurance.csv")
    ap.add_argument("--aggregate-csv", default="outputs/phase28/p28_aggregate.csv")
    ap.add_argument("--outdir", default="outputs/phase28")
    ap.add_argument("--prefix", default="p28")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.endurance_csv)

    # bar: no-slip % by target
    by = df.groupby("target")["no_slip"].apply(lambda s:(s=="yes").mean()*100).reset_index(name="pct_no_slip")
    plt.figure()
    plt.bar(by["target"], by["pct_no_slip"])
    plt.xlabel("target"); plt.ylabel("% no-slip"); plt.title("No-slip rate by target")
    plt.savefig(outdir/(args.prefix+"_noslip_by_target.png"), dpi=160, bbox_inches="tight"); plt.close()

    # t_first_slip histogram
    tfs=pd.to_numeric(df["t_first_slip"], errors="coerce").dropna()
    if len(tfs):
        plt.figure()
        tfs.hist(bins=20)
        plt.xlabel("time to first slip (s)"); plt.ylabel("count"); plt.title("Time-to-first-slip")
        plt.savefig(outdir/(args.prefix+"_t_first_slip_hist.png"), dpi=160, bbox_inches="tight"); plt.close()

    # effort & chatter hist
    for col, title in [("effort_L1","Actuator effort L1"), ("chatter","Chatter (sign changes)")]:
        vals=pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals):
            plt.figure()
            vals.hist(bins=20)
            plt.xlabel(col); plt.ylabel("count"); plt.title(title)
            plt.savefig(outdir/(args.prefix+f"_{col}_hist.png"), dpi=160, bbox_inches="tight"); plt.close()

if __name__=="__main__":
    main()
