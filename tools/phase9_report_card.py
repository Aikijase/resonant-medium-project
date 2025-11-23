#!/usr/bin/env python3
import argparse, csv, os, numpy as np
def load(p):
    d={"t":[], "x":[], "x_star":[], "err":[], "beta_eff":[], "energy":[]}
    with open(p) as f:
        r=csv.DictReader(f)
        for row in r:
            for k in d: d[k].append(float(row[k]))
    for k in d: d[k]=np.array(d[k],float)
    return d
def verdict(d):
    n=len(d["t"]); h=slice(n//2,None)
    p2p=float(np.max(d["x"][h])-np.min(d["x"][h])); rms=float(np.sqrt(np.mean(d["err"][h]**2)))
    b=float(np.mean(d["beta_eff"][h]))
    if p2p>10: v="Near-resonant oscillation (large amplitude)"
    elif b>0.5 and p2p<0.1: v="Over-damped correction, stable"
    elif 0.05<p2p<5 and 0.05<b<0.5: v="Adaptive tempo engaged, smoothing"
    else: v="Mixed/neutral regime"
    return p2p, rms, b, v
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("csvs", nargs="+"); a=ap.parse_args()
    print("\nPhase 9 — Report Card\n")
    print("{:<32} {:>12} {:>12} {:>12}  {}".format("run","peak_to_peak","rms_err","mean_beta","verdict"))
    for p in a.csvs:
        d=load(p); p2p,rms,b,v=verdict(d); name=os.path.splitext(os.path.basename(p))[0]
        print("{:<32} {:>12.6g} {:>12.6g} {:>12.6g}  {}".format(name,p2p,rms,b,v))
if __name__=="__main__": main()
