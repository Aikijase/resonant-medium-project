#!/usr/bin/env python3
import argparse, pathlib, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description="Phase-29 plotting")
    ap.add_argument("--endurance-csv", default="outputs/phase29/p29_endurance.csv")
    ap.add_argument("--aggregate-csv", default="outputs/phase29/p29_aggregate.csv")
    ap.add_argument("--outdir", default="outputs/phase29")
    ap.add_argument("--prefix", default="p29")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.endurance_csv)

    # bar: no-slip % by target
    by = df.groupby("target")["no_slip"].apply(lambda s:(s=="yes").mean()*100).reset_index(name="pct_no_slip")
    plt.figure(); plt.bar(by["target"], by["pct_no_slip"])
    plt.xlabel("target"); plt.ylabel("% no-slip"); plt.title("No-slip rate by target (with integrator)")
    plt.savefig(outdir/(args.prefix+"_noslip_by_target.png"), dpi=160, bbox_inches="tight"); plt.close()

    # histos
    for col, title in [("t_first_slip","Time-to-first-slip (s)"),
                       ("effort_L1", "Actuator effort L1"),
                       ("chatter",   "Chatter (sign changes)"),
                       ("mean_abs_err","Mean |e|")]:
        vals=pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals):
            plt.figure(); vals.hist(bins=20)
            plt.xlabel(col); plt.ylabel("count"); plt.title(title)
            plt.savefig(outdir/(args.prefix+f"_{col}_hist.png"), dpi=160, bbox_inches="tight"); plt.close()

if __name__=="__main__":
    main()
