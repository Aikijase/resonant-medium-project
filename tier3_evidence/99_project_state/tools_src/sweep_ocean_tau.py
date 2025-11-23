#!/usr/bin/env python3
import argparse, itertools, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("ERROR:", " ".join(cmd), file=sys.stderr)
        print(r.stdout, r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv", default="outputs/phase2/fs8_eval.csv")
    ap.add_argument("--out-dir", default="outputs/phase7/sweeps_tau")
    ap.add_argument("--mu0", type=float, default=0.99)
    ap.add_argument("--nu0", type=float, default=0.055)
    ap.add_argument("--tau", default="0.05,0.10,0.20,0.30")
    ap.add_argument("--kappa", default="0.01,0.02,0.03")
    ap.add_argument("--z-min", type=float, default=0.15)
    ap.add_argument("--z-max", type=float, default=1.20)
    args = ap.parse_args()

    tau_list   = [float(x) for x in args.tau.split(",")]
    kappa_list = [float(x) for x in args.kappa.split(",")]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows=[]

    for tau,k in itertools.product(tau_list, kappa_list):
        tag=f"tau{tau:.3f}_kap{k:.3f}".replace(".","p")
        fs8=Path(args.out_dir)/f"fs8_eval_ocean_{tag}.csv"
        csv=Path(args.out_dir)/f"resonant_response_{tag}.csv"
        json=Path(args.out_dir)/f"resonant_response_{tag}.json"
        png=Path(args.out_dir)/f"resonant_response_{tag}.png"

        run(["python3","tools/phase7_ocean/ode_growth_ocean_tau.py",
             "--ref-csv",args.ref_csv,"--out-csv",str(fs8),
             "--mu0",str(args.mu0), "--nu0",str(args.nu0),
             "--tau",str(tau), "--kappa",str(k)])

        run(["python3","tools/phase6_resonant_response_v5.py",
             "--fs8-csv",str(fs8),
             "--z-col","z","--lcdm-col","fs8_fit","--model-col","fs8_pred_ocean_tau",
             "--z-min",str(args.z_min),"--z-max",str(args.z_max),
             "--max-gap","0.05","--eps","0.0","--floor-frac","0.0",
             "--clip-min","0.6","--clip-max","1.4",
             "--out-csv",str(csv),"--out-json",str(json),"--out-png",str(png)])

        d=pd.read_csv(csv); R=d["R_res"].to_numpy(); m=np.isfinite(R)
        inside=int(((R[m]>=0.6)&(R[m]<=1.4)).sum()); total=int(m.sum())
        med=float(np.nanmedian(R)); q10,q90=[float(x) for x in np.nanquantile(R,[.1,.9])]
        rows.append({"tau":tau,"kappa":k,"inside":inside,"total":total,"median_R":med,
                     "q10_R":q10,"q90_R":q90,"csv":str(csv),"png":str(png)})

    df=pd.DataFrame(rows).sort_values(["inside", "median_R"], ascending=[False, True])
    out=Path(args.out_dir)/"summary_tau.csv"; df.to_csv(out, index=False)
    print(df.head(8).to_string(index=False))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
