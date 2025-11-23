#!/usr/bin/env python3
import argparse, csv, json, pathlib, subprocess, sys
from typing import List, Tuple, Optional

PY=sys.executable

def run_si(w2,K,steps=12000,burn_in=250):
    p=subprocess.run([PY,"tools/phase10_phasecouple_demo.py","--preset","neuron",
        "--omega2",str(w2),"--kv","0.10","--kx","0.20","--Kphi",str(K),
        "--eps","0.06","--adapt_every","15","--noise","0.01",
        "--steps",str(steps),"--burn_in",str(burn_in),"--prefix",f"fill_w{w2}_K{K}"],
        capture_output=True,text=True,check=True)
    return float(json.loads(p.stdout)["metrics"]["sync_index"])

def coarse_scan(w2,Kmin,Kmax,n,target,steps,burn):
    xs=[Kmin+i*(Kmax-Kmin)/(n-1) for i in range(n)]
    vals=[run_si(w2,x,steps,burn) for x in xs]
    br=[]
    for (K0,s0),(K1,s1) in zip(zip(xs,vals), zip(xs[1:],vals[1:])):
        d0,d1=s0-target,s1-target
        if d0==0: br.append((K0,K0))
        elif d0*d1<0: br.append((K0,K1))
    return br

def bisect(w2,a,b,target,tol,maxit,steps,burn):
    import math
    def f(K): return run_si(w2,K,steps,burn)-target
    fa,fb=f(a),f(b)
    if fa==0: return a
    if fb==0: return b
    if fa*fb>0: return None
    for _ in range(maxit):
        m=0.5*(a+b); fm=f(m)
        if abs(fm)<=tol or abs(b-a)<=max(tol,1e-6): return m
        if fa*fm<=0: b,fb=m,fm
        else: a,fa=m,fm
    return 0.5*(a+b)

def solve_two(w2,target,Kmin,Kmax,coarse_n,tol,maxit,steps,burn):
    # try wide, dense coarse scan to catch skinny bands
    br=coarse_scan(w2,Kmin,Kmax,coarse_n,target,steps,burn)
    if not br:
        span=Kmax-Kmin
        br=coarse_scan(w2,max(0.0,Kmin-0.5*span),Kmax+0.5*span,coarse_n,target,steps,burn)
        if not br: return (None,None)
    # unique + left→right
    uniq=[]
    for a,b in br:
        if not uniq or abs(a-uniq[-1][0])>1e-6 or abs(b-uniq[-1][1])>1e-6:
            uniq.append((a,b))
    uniq.sort(key=lambda ab:0.5*(ab[0]+ab[1]))
    roots=[]
    for a,b in uniq[:2]:
        r=bisect(w2,a,b,target,tol,maxit,steps,burn)
        if r is not None: roots.append(r)
    roots.sort()
    if not roots: return (None,None)
    if len(roots)==1:
        mid=0.5*(Kmin+Kmax)
        return (roots[0], None) if roots[0]<=mid else (None, roots[0])
    return (roots[0], roots[-1])

def append_points(out_csv,out_json,rows):
    new_file=not out_csv.exists()
    with out_csv.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["target","omega2","Kphi_lo","Kphi_hi"])
        if new_file: w.writeheader()
        for r in rows: w.writerow(r)
    # rebuild json sorted
    acc=[]
    with out_csv.open(newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            acc.append({"target": float(row["target"]), "omega2": float(row["omega2"]),
                        "Kphi_lo": float(row["Kphi_lo"]), "Kphi_hi": float(row["Kphi_hi"])})
    acc=sorted({(x["target"],x["omega2"]):x for x in acc}.values(), key=lambda x:(x["target"],x["omega2"]))
    with out_json.open("w") as f: json.dump(acc,f,indent=2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",default="outputs/phase21")
    ap.add_argument("--targets",required=True,help="e.g. 0.90,0.94")
    ap.add_argument("--w2",required=True,help="e.g. 2.82,2.84,2.86")
    ap.add_argument("--Kmin",type=float,default=0.30)
    ap.add_argument("--Kmax",type=float,default=1.80)
    ap.add_argument("--coarse-n",type=int,default=41)
    ap.add_argument("--steps",type=int,default=12000)
    ap.add_argument("--burn-in",type=int,default=250)
    ap.add_argument("--tol",type=float,default=5e-4)
    ap.add_argument("--max-iters",type=int,default=60)
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    out_csv=outdir/"p21_guardbands_points.csv"
    out_json=outdir/"p21_guardbands_points.json"

    targets=[float(x) for x in args.targets.split(",") if x.strip()]
    w2s=[float(x) for x in args.w2.split(",") if x.strip()]

    added=[]
    for t in targets:
        for w2 in w2s:
            lo,hi=solve_two(w2,t,args.Kmin,args.Kmax,args.coarse_n,args.tol,args.max_iters,args.steps,args.burn_in)
            if lo is None and hi is None:
                continue
            if lo is None:
                mid=0.5*(args.Kmin+args.Kmax); lo=max(args.Kmin,min(args.Kmax,2*mid-hi))
            if hi is None:
                mid=0.5*(args.Kmin+args.Kmax); hi=max(args.Kmin,min(args.Kmax,2*mid-lo))
            added.append({"target":t,"omega2":w2,"Kphi_lo":float(lo),"Kphi_hi":float(hi)})

    if added:
        append_points(out_csv,out_json,added)
    print(json.dumps({"status":"ok","added":len(added),"out_csv":str(out_csv),"out_json":str(out_json)},indent=2))

if __name__=="__main__":
    main()
