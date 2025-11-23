#!/usr/bin/env python3
import argparse, json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def law(oh,A,w0,p): return A*(1-np.exp(- (oh/w0)**p ))

def load_norm(paths):
    out=[]
    for p in paths:
        df=pd.read_csv(p)
        df=df.rename(columns=str.strip)
        for k in ["omega","omega0","epsilon"]:
            if k not in df: raise SystemExit(f"{p} missing {k}")
        df["omega_hat"]=df["omega"]/df["omega0"]
        if "label" not in df: df["label"]=os.path.basename(p)
        df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=["omega_hat","epsilon"])
        df["epsilon"]=df["epsilon"].clip(0,1)
        out.append(df[["omega_hat","epsilon","label"]])
    return pd.concat(out, ignore_index=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-prefix", required=True)
    args=ap.parse_args()

    df=load_norm(args.inputs)
    x=df["omega_hat"].to_numpy(float); y=df["epsilon"].to_numpy(float)
    popt,_=curve_fit(law, x, y, p0=[1,1,1], maxfev=200000)
    A,w0,p=map(float,popt); pred=law(x,*popt)
    rms=float(np.sqrt(np.mean((y-pred)**2)))
    span=y.max()-y.min() if y.max()>y.min() else 1.0
    nrmse=float(rms/span)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    fig,ax=plt.subplots()
    for lab,sub in df.groupby("label"):
        ax.scatter(sub["omega_hat"], sub["epsilon"], s=12, label=lab)
    gx=np.linspace(max(1e-8,x.min()), x.max(), 600)
    ax.plot(gx, law(gx,*popt), linewidth=2)
    ax.set_xlabel("ω̂=ω/ω0"); ax.set_ylabel("ε"); ax.set_title("ε(ω̂) collapse"); ax.legend()
    fig.tight_layout(); fig.savefig(args.out_prefix+".png", dpi=200); plt.close(fig)

    meta={"fit_params":{"A":A,"w0":w0,"p":p},"metrics":{"rms":rms,"nrmse":nrmse},"inputs":args.inputs}
    with open(args.out_prefix+".json","w") as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta, indent=2))
if __name__=="__main__": main()
