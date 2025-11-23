#!/usr/bin/env python3
import csv, sys, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

def load_rows(p):
    rows=[]
    with open(p) as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            try:
                rows.append(dict(
                    bridges=int(r["bridges"]),
                    noise=float(r["noise"]),
                    alpha=float(r["alpha"]),
                    seed=int(r["seed"]),
                    R=float(r["R_mean"]),
                    span=float(r["phase_span"]),
                    ivar=float(r["inter_lag_var"]),
                    K=float(r["K_edge"]),
                    l2=float(r.get("lambda2",0.0)),
                ))
            except: pass
    return rows

def mean_by_alpha(rows, key):
    d=defaultdict(list)
    for r in rows: d[r["alpha"]].append(r[key])
    al=sorted(d); vals=[float(np.mean(d[a])) for a in al]
    return al, vals

def alpha_star(rows, thr):
    al,R = mean_by_alpha(rows,"R")
    for a, r in zip(al,R):
        if r >= thr: return a, r
    return None, None

def main():
    if len(sys.argv)<3:
        print("usage: phase12_analyze.py <csv> <out_prefix>"); sys.exit(1)
    csvp, prefix = sys.argv[1:3]
    rows=load_rows(csvp)
    if not rows: sys.exit("No rows found")

    # α* table by (bridges, noise)
    combos=sorted({(r["bridges"], r["noise"]) for r in rows})
    thr_list=[0.90,0.95,0.98]
    lines=["bridges,noise,alpha*_0.90,alpha*_0.95,alpha*_0.98"]
    for (br,nz) in combos:
        subset=[r for r in rows if r["bridges"]==br and r["noise"]==nz]
        astars=[]
        for thr in thr_list:
            a, _ = alpha_star(subset, thr)
            astars.append("" if a is None else f"{a:.2f}")
        lines.append(f"{br},{nz}," + ",".join(astars))
    with open(f"{prefix}_alpha_star.csv","w") as f:
        f.write("\n".join(lines))
    print(f"[p12] wrote {prefix}_alpha_star.csv")

    # R vs alpha (lines by bridges) per noise
    noises=sorted({r["noise"] for r in rows})
    for nz in noises:
        sub=[r for r in rows if r["noise"]==nz]
        byB=defaultdict(list)
        for r in sub: byB[r["bridges"]].append(r)
        # R vs α
        plt.figure()
        for br, grp in sorted(byB.items()):
            al, y = mean_by_alpha(grp, "R")
            plt.plot(al, y, marker='o', label=f"bridges={br}")
        plt.xlabel("alpha"); plt.ylabel("R_mean"); plt.title(f"R vs alpha (noise={nz})")
        plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(f"{prefix}_R_vs_alpha_n{nz}.png", dpi=150); plt.close()
        # inter-lag var vs α
        plt.figure()
        for br, grp in sorted(byB.items()):
            al, y = mean_by_alpha(grp, "ivar")
            plt.plot(al, y, marker='o', label=f"bridges={br}")
        plt.xlabel("alpha"); plt.ylabel("inter_lag_var (↓ better)")
        plt.title(f"Inter-module lag variance (noise={nz})")
        plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(f"{prefix}_ivar_vs_alpha_n{nz}.png", dpi=150); plt.close()

    # NEW: R vs lambda2 scatter per noise (colors=bridges)
    for nz in noises:
        sub=[r for r in rows if r["noise"]==nz]
        plt.figure()
        for br in sorted({r["bridges"] for r in sub}):
            grp=[r for r in sub if r["bridges"]==br]
            x=[r["l2"] for r in grp]; y=[r["R"] for r in grp]
            plt.scatter(x,y,label=f"bridges={br}", s=30)
        plt.xlabel("lambda2 (algebraic connectivity)"); plt.ylabel("R_mean")
        plt.title(f"R vs λ2 (noise={nz})")
        plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(f"{prefix}_R_vs_lambda2_n{nz}.png", dpi=150); plt.close()

    print(f"[p12] wrote plots with prefix {prefix}_*")
if __name__=="__main__":
    main()
