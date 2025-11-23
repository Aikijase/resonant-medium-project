#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_rows(csvp):
    rows=[]
    with open(csvp) as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            try:
                rows.append(dict(
                    M=int(r["M"]), n=int(r["n"]),
                    intra=r["intra"],
                    bridge_mode=r["bridge_mode"],
                    bridges=int(r["bridges"]),
                    alpha=float(r["alpha"]),
                    noise=float(r["noise"]),
                    seed=int(r["seed"]),
                    R=float(r["R_mean"]),
                    P=float(r["pair_coh"]),
                    K=float(r["K_edge"]),
                    span=float(r["phase_span"]),
                    cvar=float(r["circ_var"]),
                    ivar=float(r["inter_lag_var"]),
                    Rmods=[float(r.get(f"R_mod{i}",0.0)) for i in range(int(r["M"]))]
                ))
            except Exception:
                pass
    return rows

def mean_by(group, key):
    return float(np.mean([x[key] for x in group]))

def plot_sets(rows, prefix):
    Path(Path(prefix).parent).mkdir(parents=True, exist_ok=True)
    # R vs alpha, one panel per noise, lines by bridges
    noises=sorted(set(r["noise"] for r in rows))
    for nz in noises:
        subset=[r for r in rows if r["noise"]==nz]
        byB=defaultdict(list)
        for r in subset: byB[r["bridges"]].append(r)
        plt.figure()
        for br, grp in sorted(byB.items()):
            alphas=sorted(set(r["alpha"] for r in grp))
            y=[np.mean([x["R"] for x in grp if x["alpha"]==a]) for a in alphas]
            plt.plot(alphas,y,marker='o',label=f"bridges={br}")
        plt.xlabel("alpha"); plt.ylabel("R_mean"); plt.title(f"R vs alpha (noise={nz})")
        plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        p1=f"{prefix}_R_n{nz}.png"; plt.savefig(p1,dpi=150); plt.close()

        # span vs alpha
        plt.figure()
        for br, grp in sorted(byB.items()):
            alphas=sorted(set(r["alpha"] for r in grp))
            y=[np.mean([x["span"] for x in grp if x["alpha"]==a]) for a in alphas]
            plt.plot(alphas,y,marker='o',label=f"bridges={br}")
        plt.xlabel("alpha"); plt.ylabel("phase_span (rad)")
        plt.title(f"Phase span vs alpha (noise={nz})")
        plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        p2=f"{prefix}_span_n{nz}.png"; plt.savefig(p2,dpi=150); plt.close()

        # inter-module lag variance vs alpha
        plt.figure()
        for br, grp in sorted(byB.items()):
            alphas=sorted(set(r["alpha"] for r in grp))
            y=[np.mean([x["ivar"] for x in grp if x["alpha"]==a]) for a in alphas]
            plt.plot(alphas,y,marker='o',label=f"bridges={br}")
        plt.xlabel("alpha"); plt.ylabel("inter_lag_var")
        plt.title(f"Inter-module lag variance vs alpha (noise={nz})")
        plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        p3=f"{prefix}_ivar_n{nz}.png"; plt.savefig(p3,dpi=150); plt.close()

    # Per-module R bars at a representative setting (pick max alpha, median bridges)
    if rows:
        A=sorted(set(r["alpha"] for r in rows)); a=A[-1]
        B=sorted(set(r["bridges"] for r in rows)); b=B[min(len(B)//2, len(B)-1)]
        nz=min(noises)
        sel=[r for r in rows if r["alpha"]==a and r["bridges"]==b and r["noise"]==nz]
        if sel:
            m=len(sel[0]["Rmods"])
            means=np.mean([r["Rmods"] for r in sel],axis=0)
            plt.figure()
            plt.bar(np.arange(m),means)
            plt.xlabel("module idx"); plt.ylabel("R_module")
            plt.title(f"Per-module R at alpha={a}, bridges={b}, noise={nz}")
            plt.tight_layout(); p4=f"{prefix}_Rmods.png"; plt.savefig(p4,dpi=150); plt.close()

    print(f"[p12] wrote plots with prefix {prefix}")
    return True

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="outputs/phase12/data/multichord_sweep.csv")
    ap.add_argument("--prefix",default="outputs/phase12/plots/multichord")
    args=ap.parse_args()
    rows=load_rows(args.csv)
    if not rows: raise SystemExit(f"No rows in {args.csv}")
    plot_sets(rows,args.prefix)
