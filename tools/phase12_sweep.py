#!/usr/bin/env python3
import csv
from pathlib import Path
from phase12_multichord_sim import run_sim

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=3)
    ap.add_argument("--n",type=int,default=6)
    ap.add_argument("--centers",nargs="+",type=float,default=[2.7,2.8,2.9])
    ap.add_argument("--span",type=float,default=0.24)
    ap.add_argument("--patterns",nargs="+",default=["block","grad","alt"])
    ap.add_argument("--intra",choices=["ring","chain","clique"],default="ring")
    ap.add_argument("--bridge_mode",choices=["none","ring","star","random_k"],default="random_k")
    ap.add_argument("--bridges",nargs="+",type=int,default=[0,1,2])
    ap.add_argument("--alpha",nargs="+",type=float,default=[0.5,1.0,1.5])
    ap.add_argument("--noise",nargs="+",type=float,default=[0.0,0.02])
    ap.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--dt",type=float,default=0.01)
    ap.add_argument("--burn_in",type=int,default=2000)
    ap.add_argument("--sample_every",type=int,default=10)
    ap.add_argument("--tail",type=int,default=500)
    ap.add_argument("--fit",default="outputs/phase10/tongue_master_wide.fit.json")
    ap.add_argument("--out",default="outputs/phase12/data/multichord_sweep.csv")
    args=ap.parse_args()

    Path("outputs/phase12/data").mkdir(parents=True, exist_ok=True)

    rows=[]
    for br in args.bridges:
        for a in args.alpha:
            for nz in args.noise:
                for sd in args.seeds:
                    res=run_sim(M=args.M,n=args.n,centers=tuple(args.centers),span=args.span,
                                patterns=tuple(args.patterns),intra=args.intra,
                                bridge_mode=args.bridge_mode,bridges=br,
                                alpha=a,steps=args.steps,dt=args.dt,burn_in=args.burn_in,
                                sample_every=args.sample_every,tail=args.tail,noise=nz,seed=sd,
                                fit_path=args.fit)
                    row=dict(M=args.M,n=args.n,intra=args.intra,
                             centers=",".join(map(str,args.centers)),
                             patterns=",".join(args.patterns),
                             bridge_mode=args.bridge_mode,bridges=br,
                             alpha=a,noise=nz,seed=sd,
                             R_mean=res["R_mean"],pair_coh=res["pair_coh"],K_edge=res["K_edge"],
                             phase_span=res["phase_span"],circ_var=res["circ_var"],
                             inter_lag_var=res["inter_lag_var"],lambda2=res.get("lambda2",0.0))
                    for i,rm in enumerate(res["R_modules"]):
                        row[f"R_mod{i}"]=rm
                    rows.append(row)
                    print(f"[p12] br={br} α={a:.2f} noise={nz:.3f} seed={sd} -> "
                          f"R={res['R_mean']:.3f} λ2={row['lambda2']:.4f} "
                          f"ivar={row['inter_lag_var']:.3f} span={row['phase_span']:.3f} K≈{row['K_edge']:.3f}")

    mod_cols=[f"R_mod{i}" for i in range(args.M)]
    cols=["M","n","intra","centers","patterns","bridge_mode","bridges","alpha","noise","seed",
          "R_mean","pair_coh","K_edge","phase_span","circ_var","inter_lag_var","lambda2"]+mod_cols
    with open(args.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[p12] wrote: {args.out}")

if __name__=="__main__":
    main()
