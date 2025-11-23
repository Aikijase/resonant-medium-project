#!/usr/bin/env python3
import json, argparse, numpy as np, pandas as pd
from scipy.optimize import curve_fit

def law(oh,A,w0,p):
    oh=np.clip(oh,1e-12,np.inf); w0=max(float(w0),1e-12)
    return A*(1-np.exp(-(oh/w0)**p))

def load(path):
    df=pd.read_csv(path)
    if "omega_hat" in df.columns: oh=df["omega_hat"].to_numpy(float)
    else: oh=(df["omega"]/df["omega0"]).to_numpy(float)
    y=df["epsilon"].to_numpy(float)
    m=np.isfinite(oh)&np.isfinite(y)
    lab=df["label"].iloc[0] if "label" in df.columns and len(df)>0 else path
    return pd.DataFrame({"omega_hat":oh[m],"epsilon":np.clip(y[m],0,1),"label":lab})

ap=argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--out", default="outputs/lodo_sb.json")
args=ap.parse_args()

D=pd.concat([load(p) for p in args.inputs], ignore_index=True)
rows=[]
for hold in sorted(D["label"].unique()):
    TR=D[D["label"]!=hold]; TE=D[D["label"]==hold]
    counts=TR.groupby("label").size().to_dict()
    sigma=TR["label"].map({k:np.sqrt(v) for k,v in counts.items()}).to_numpy(float)
    (A,w0,p),_=curve_fit(law,TR["omega_hat"],TR["epsilon"],
                         p0=[1,1,1],sigma=sigma,bounds=([0,1e-3,0.1],[1.2,10,8]),maxfev=200000)
    y=TE["epsilon"].to_numpy(float)
    pred=law(TE["omega_hat"].to_numpy(float),A,w0,p)
    rms=float(np.sqrt(np.mean((y-pred)**2))); span=y.max()-y.min() or 1.0
    rows.append({"held_out":hold,"A":float(A),"w0":float(w0),"p":float(p),
                 "nrmse":float(rms/span),"n":int(len(y))})
res={"lodo":rows}
print(json.dumps(res,indent=2))
with open(args.out,"w") as f: json.dump(res,f,indent=2)
