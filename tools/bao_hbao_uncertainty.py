#!/usr/bin/env python3
"""
BAO hBAO Uncertainty + Robustness Runner
----------------------------------------
Runs:
  1) Ultrafine hBAO sweep → best + 1σ
  2) Leave-One-Out (LOOCV) robustness
  3) H0 sensitivity sweep

Assumes:
- DESI long-form BAO data (kind,z,y_data,sigma)
- Global BAO scaling (no tracer split)
"""

import argparse, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ENGINE = Path("tools/bao_tracer_sweep.py")

def run(cmd):
    env = os.environ.copy(); env.setdefault("PYTHONPATH", ".")
    cp = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if cp.returncode != 0:
        print(cp.stdout); print(cp.stderr)
        raise RuntimeError("Command failed")
    return cp.stdout

def parse_best(stdout):
    best = {}
    for line in stdout.splitlines():
        if line.startswith("Best:"):
            parts = line.split()
            for p in parts:
                if "=" in p:
                    k,v = p.split("=",1)
                    try: best[k]=float(v)
                    except: pass
    return best

def ultrafine(args):
    out = f"{args.out_dir}/hbao_ultrafine"
    cmd = [
        sys.executable, str(ENGINE),
        "--bao-csv", args.bao_csv,
        "--cov-csv", args.cov_csv,
        "--om0", str(args.om0), "--h", str(args.h),
        "--z-col", args.z_col, "--y-col", args.y_col, "--kind-col", args.kind_col,
        "--hELG-range", str(args.hbao_min), str(args.hbao_max), str(args.hbao_steps),
        "--hLRG2-range", "1.00", "1.00", "1",
        "--elg-pattern", ".*", "--lrg2-pattern", "$^",
        "--out-prefix", out
    ]
    sout = run(cmd)
    best = parse_best(sout)
    return best.get("hELG"), best.get("chi2"), best.get("dof"), Path(out + "_summary.csv")

def one_sigma(csv_path):
    df = pd.read_csv(csv_path)
    i0 = df.chi2.idxmin()
    h0 = df.hELG[i0]; chi0 = df.chi2[i0]
    mask = df.chi2 <= chi0+1
    low = df[mask & (df.hELG<=h0)].hELG.max()
    high= df[mask & (df.hELG>=h0)].hELG.min()
    return h0, chi0, low, high

def loocv(args):
    src = Path(args.bao_csv)
    rows = sum(1 for _ in open(src)) - 1
    rec=[]
    for i in range(1,rows+1):
        tmp = Path(f"/tmp/bao_loocv_{i}.csv")
        with open(src) as fin, open(tmp,"w") as fout:
            header=fin.readline(); fout.write(header)
            for j,line in enumerate(fin,1):
                if j==i: continue
                fout.write(line)
        cmd = [
            sys.executable,str(ENGINE),
            "--bao-csv",str(tmp),"--cov-csv","auto",
            "--om0",str(args.om0),"--h",str(args.h),
            "--z-col",args.z_col,"--y-col",args.y_col,"--kind-col",args.kind_col,
            "--hELG-range",str(args.hbao_min),str(args.hbao_max),"41",
            "--hLRG2-range","1.00","1.00","1",
            "--elg-pattern",".*","--lrg2-pattern","$^",
            "--out-prefix",f"{args.out_dir}/loocv_{i}"
        ]
        sout = run(cmd); best=parse_best(sout)
        rec.append({"drop_row":i,"hBAO":best.get("hELG"),"chi2":best.get("chi2")})
    return pd.DataFrame(rec)

def h0_sense(args,hbao):
    rec=[]
    for HH in args.h0_list:
        cmd=[
            sys.executable,str(ENGINE),
            "--bao-csv",args.bao_csv,"--cov-csv",args.cov_csv,
            "--om0",str(args.om0),"--h",str(HH),
            "--z-col",args.z_col,"--y-col",args.y_col,"--kind-col",args.kind_col,
            "--hELG-fixed",str(hbao),"--hLRG2-fixed","1.00",
            "--elg-pattern",".*","--lrg2-pattern","$^",
            "--out-prefix",f"{args.out_dir}/hbao_h{str(HH).replace('.','')}"
        ]
        sout=run(cmd); best=parse_best(sout)
        rec.append({"H0":HH,"chi2":best.get("chi2")})
    return pd.DataFrame(rec)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bao-csv",default="data/desi_dr1_bao/bao_measurements_long_sigma.csv")
    p.add_argument("--cov-csv",default="data/desi_dr1_bao/bao_covariance.csv")
    p.add_argument("--om0",type=float,default=0.320)
    p.add_argument("--h",type=float,default=0.70)
    p.add_argument("--hbao-min",type=float,default=0.964)
    p.add_argument("--hbao-max",type=float,default=0.976)
    p.add_argument("--hbao-steps",type=int,default=121)
    p.add_argument("--z-col",default="z")
    p.add_argument("--y-col",default="y_data")
    p.add_argument("--kind-col",default="kind")
    p.add_argument("--h0-list",nargs="*",type=float,default=[0.68,0.70,0.72])
    p.add_argument("--out-dir",default="outputs/thrace_best")
    args=p.parse_args()

    Path(args.out_dir).mkdir(parents=True,exist_ok=True)

    hbest,chi2_best,dof,csv_path = ultrafine(args)
    h0,chi0,low,high = one_sigma(csv_path)
    loocv_df = loocv(args); loocv_df.to_csv(f"{args.out_dir}/loocv_summary.csv",index=False)
    h0df = h0_sense(args,h0); h0df.to_csv(f"{args.out_dir}/h0_sensitivity_summary.csv",index=False)

    with open(f"{args.out_dir}/bao_hbao_uncertainty_report.txt","w") as f:
        f.write("BAO hBAO Uncertainty + Robustness (previous data only)\n")
        f.write(f"Om0={args.om0}, h={args.h}\n")
        f.write(f"hBAO = {h0:.6f} (-{h0-low:.6f}, +{high-h0:.6f})\n")
        f.write(f"chi2_best = {chi0}\n\n")
        f.write("LOOCV:\n"+loocv_df.to_string(index=False)+"\n\n")
        f.write("H0 sensitivity:\n"+h0df.to_string(index=False)+"\n")

if __name__=="__main__":
    main()
