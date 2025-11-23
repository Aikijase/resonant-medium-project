#!/usr/bin/env python3
import argparse, pathlib, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def heatmap(df, out_png, title):
    Rs=sorted(df["R"].unique())
    Hs=sorted(df["deadband_frac"].unique())
    M=np.full((len(Hs), len(Rs)), np.nan)
    for i,h in enumerate(Hs):
        for j,R in enumerate(Rs):
            sub=df[(df["R"]==R)&(df["deadband_frac"]==h)]
            if len(sub): M[i,j]=sub["pct_no_slip"].mean()
    plt.figure()
    plt.imshow(M, aspect="auto", origin="lower",
               extent=[min(Rs), max(Rs), min(Hs), max(Hs)])
    plt.colorbar(label="% no-slip")
    plt.xlabel("R (max dK_phi/dt)")
    plt.ylabel("deadband_frac (f_h)")
    plt.title(title)
    plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()

def main():
    ap=argparse.ArgumentParser(description="Phase-27 plotting")
    ap.add_argument("--policy-map-csv", default="outputs/phase27/p27_policy_map.csv")
    ap.add_argument("--outdir", default="outputs/phase27")
    ap.add_argument("--prefix", default="p27")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.policy_map_csv)

    # per-target heatmaps
    for tgt, g in df.groupby("target"):
        out=outdir/(args.prefix+f"_t{tgt:.2f}_policy_heatmap.png")
        heatmap(g, out, f"Policy map @ target {tgt:.2f}")

    # best row (for overlay text table)
    best=df.sort_values(["pct_no_slip","R","deadband_frac"], ascending=[False,True,False]).head(1)
    txt = (best[["target","R","deadband_frac","pct_no_slip","v","w2_start","w2_end"]]).to_string(index=False)
    fig=plt.figure()
    plt.axis("off")
    plt.text(0.01,0.99, "Recommended policy\n" + txt, va="top", ha="left")
    out2=outdir/(args.prefix+"_recommendation.png")
    plt.savefig(out2, dpi=160, bbox_inches="tight"); plt.close()

if __name__=="__main__":
    main()
