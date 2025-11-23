#!/usr/bin/env python3
"""
Phase-6 (v4): Resonant Response with z-mask + data-driven floor.
"""
import os, json, math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def interp_over_gaps(x, y, max_gap=None):
    x, y = np.asarray(x,float), np.asarray(y,float)
    y2 = y.copy(); n_fix = n_left = 0; i=0; N=len(y2)
    while i<N:
        if np.isfinite(y2[i]): i+=1; continue
        j=i
        while j<N and not np.isfinite(y2[j]): j+=1
        l=i-1; r=j
        if l>=0 and r<N and np.isfinite(y2[l]) and np.isfinite(y2[r]):
            gap=x[r]-x[l]
            if max_gap is None or gap<=max_gap:
                for k in range(i,j):
                    t=(x[k]-x[l])/(x[r]-x[l])
                    y2[k]=y2[l]+t*(y2[r]-y2[l])
                n_fix+=j-i
            else: n_left+=j-i
        i=j
    return y2,n_fix,n_left

def trimmed_stats(a,lo=0.1,hi=0.9):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not a.size: return {"mean":math.nan,"std":math.nan,"n":0}
    ql,qh=np.quantile(a,[lo,hi]); s=a[(a>=ql)&(a<=qh)]
    return {"mean":float(s.mean()),"std":float(s.std()),"n":int(len(s))}

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--fs8-csv",default="outputs/phase2/fs8_eval.csv")
    ap.add_argument("--z-col",default="z")
    ap.add_argument("--lcdm-col",default="fs8_fit")
    ap.add_argument("--model-col",default="fs8_pred")
    ap.add_argument("--alpha",type=float,default=0.0)
    ap.add_argument("--max-gap",type=float,default=0.10)
    ap.add_argument("--floor-frac",type=float,default=0.0)
    ap.add_argument("--floor-percentile",type=float,default=None)
    ap.add_argument("--z-min",type=float,default=None)
    ap.add_argument("--z-max",type=float,default=None)
    ap.add_argument("--clip-min",type=float,default=0.6)
    ap.add_argument("--clip-max",type=float,default=1.4)
    ap.add_argument("--out-csv",default="outputs/phase6/resonant_response.csv")
    ap.add_argument("--out-json",default="outputs/phase6/resonant_response.json")
    ap.add_argument("--out-png",default="plots/phase6/resonant_response.png")
    args=ap.parse_args()

    Path("outputs/phase6").mkdir(parents=True,exist_ok=True)
    Path("plots/phase6").mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.fs8_csv)
    z=df[args.z_col].to_numpy(float)
    L=df[args.lcdm_col].to_numpy(float)
    M=df[args.model_col].to_numpy(float)

    mask=np.isfinite(z)
    if args.z_min is not None: mask&=(z>=args.z_min)
    if args.z_max is not None: mask&=(z<=args.z_max)
    z,L,M=z[mask],L[mask],M[mask]

    L=np.where((L<=0)|~np.isfinite(L),np.nan,L)
    M=np.where((M<=0)|~np.isfinite(M),np.nan,M)

    Lr,Lf,Ll=interp_over_gaps(z,L,args.max_gap)
    Mr,Mf,Ml=interp_over_gaps(z,M,args.max_gap)

    posM=Mr[np.isfinite(Mr)&(Mr>0)]
    # ----- floor guards -----
    floor_val = 0.0
    # optional absolute floor: fraction of median(L)
    medL = np.nanmedian(L_rep)
    if args.floor_frac > 0 and np.isfinite(medL):
        floor_val = max(floor_val, args.floor_frac * medL)
    # optional percentile floor of positive model (rarely needed now)
    if args.floor_percentile is not None:
        posM = M_rep[np.isfinite(M_rep) & (M_rep > 0)]
        if posM.size:
            floor_val = max(floor_val, float(np.percentile(posM, args.floor_percentile)))

    # NEW: local, science-clean floor — scales with L(z)
    eps = 0.02  # 2% of L(z); small enough to avoid bias, big enough to stop blow-ups
    local_floor = eps * L_rep
    if floor_val > 0:
        local_floor = np.maximum(local_floor, floor_val)

    M_rep = np.where(np.isfinite(M_rep), np.maximum(M_rep, local_floor), np.nan)
    stats=trimmed_stats(R[fin])

    out=df.loc[mask].copy()
    out["fs8_lcdm_repaired"]=Lr
    out["fs8_model_repaired"]=Mr
    out["R_res"]=R
    out.to_csv(args.out_csv,index=False)

    meta={"z_window":[args.z_min,args.z_max],
          "repairs":{"lcdm_fixed":Lf,"lcdm_left":Ll,
                     "model_fixed":Mf,"model_left":Ml},
          "stats_trimmed_10_90":stats}
    json.dump(meta,open(args.out_json,"w"),indent=2)
    print(f"Wrote {args.out_csv} and {args.out_json}")

    m=fin&(R>args.clip_min)&(R<args.clip_max)
    plt.figure(figsize=(6.4,4.2))
    plt.axhline(1,ls="--",lw=1)
    plt.plot(z[m],R[m],"o-",label="R(z)")
    plt.xlabel("z"); plt.ylabel("R(z)")
    plt.title("Phase-6 Resonant Response (v4)")
    plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(args.out_png,dpi=160)
    print(f"Wrote {args.out_png}")

if __name__=="__main__":
    main()
