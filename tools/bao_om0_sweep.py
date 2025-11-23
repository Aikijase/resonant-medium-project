#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, math

from bao_stack_autodetect import read_wide, build_block, LCDM_dist

def chi2_for(z,kinds,y,C,Om0,h):
    DM,DH,DV = LCDM_dist(z,Om0,h)
    m=[]; iDM=iDH=iDV=0
    for k in kinds:
        if k=="DM": m.append(DM[iDM]); iDM+=1
        elif k=="DH": m.append(DH[iDH]); iDH+=1
        elif k=="DV": m.append(DV[iDV]); iDV+=1
    m=np.array(m); y=np.array(y)
    eps=1e-12*np.median(np.diag(C)); Cf=C+eps*np.eye(C.shape[0])
    Ci=np.linalg.inv(Cf)
    beta=float((m@(Ci@y))/max(m@(Ci@m),1e-300))
    r=y-beta*m
    return float(r@(Ci@r)), beta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bao-csv", default="bao_measurements.csv")
    ap.add_argument("--cov-csv", default="bao_covariance.csv")
    ap.add_argument("--h", type=float, default=0.7)
    ap.add_argument("--om0-min", type=float, default=0.2)
    ap.add_argument("--om0-max", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=41)
    args=ap.parse_args()

    z,DM,DH,DV = read_wide(args.bao_csv)
    z_vec,kinds,y_vec = build_block(z,DM,DH,DV)
    C = np.loadtxt(args.cov_csv, delimiter=",").astype(float)

    grid=np.linspace(args.om0_min,args.om0_max,args.steps)
    rows=[]
    for Om0 in grid:
        c2,b=chi2_for(z_vec,kinds,y_vec,C,Om0,args.h)
        dof=len(y_vec)-1
        rows.append((Om0,c2,dof,c2/dof,b))
    df=pd.DataFrame(rows,columns=["Om0","chi2","dof","chi2_dof","beta"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    best=df.loc[df["chi2"].idxmin()]
    print("\nBest (coarse): Om0={Om0:.4f},  χ²/dof={chi2_dof:.3f},  β={beta:.6g}".format(**best.to_dict()))

if __name__=="__main__":
    main()
