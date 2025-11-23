#!/usr/bin/env python3
import argparse, json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def law(oh,A,w0,p):
    oh = np.clip(oh,1e-12,np.inf); w0=max(float(w0),1e-12)
    return A*(1.0 - np.exp(- (oh/w0)**p))

def load(paths):
    out=[]
    for p in paths:
        df=pd.read_csv(p)
        if "omega_hat" in df.columns:  # allow pre-normalized
            oh=df["omega_hat"].to_numpy(float)
        else:
            oh=(df["omega"]/df["omega0"]).to_numpy(float)
        eps=df["epsilon"].to_numpy(float)
        lab=(df["label"].iloc[0] if "label" in df.columns and len(df)>0 else os.path.basename(p))
        m=np.isfinite(oh)&np.isfinite(eps)
        out.append(pd.DataFrame({"omega_hat":oh[m],"epsilon":np.clip(eps[m],0,1),"label":lab}))
    return pd.concat(out, ignore_index=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",nargs="+",required=True)
    ap.add_argument("--out-prefix",required=True)
    ap.add_argument("--equalize",action="store_true",help="equalize dataset influence via sigma weights")
    args=ap.parse_args()

    D=load(args.inputs)
    x=D["omega_hat"].to_numpy(float); y=D["epsilon"].to_numpy(float)

    # Optional per-dataset equal weighting
    sigma=np.ones_like(y)
    if args.equalize:
        counts=D.groupby("label").size().to_dict()
        w = {k: np.sqrt(v) for k,v in counts.items()}  # sigma_i = sqrt(n_group)
        sigma = D["label"].map(w).to_numpy(float)

    # BOUNDS keep parameters physical (fixes the A/w0 blow-up)
    lb=np.array([0.0, 1e-3, 0.1])
    ub=np.array([1.2, 10.0, 8.0])

    popt,_=curve_fit(law, x, y, p0=[1.0,1.0,1.0], sigma=sigma, absolute_sigma=False,
                     bounds=(lb,ub), maxfev=200000)
    A,w0,p=map(float,popt)
    pred=law(x,*popt)
    rms=float(np.sqrt(np.mean((y-pred)**2)))
    span=y.max()-y.min() if y.max()>y.min() else 1.0
    nrmse=float(rms/span)

    # Plot
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    fig,ax=plt.subplots()
    for lab,g in D.groupby("label"):
        ax.scatter(g["omega_hat"], g["epsilon"], s=12, label=lab)
    gx=np.linspace(max(1e-4,x.min()), x.max(), 600)
    ax.plot(gx, law(gx,*popt), linewidth=2)
    ax.set_xlabel("ω̂ = ω/ω₀"); ax.set_ylabel("ε"); ax.set_title("ε(ω̂) collapse (bounded)")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out_prefix+".png", dpi=200); plt.close(fig)

    meta={"fit_params":{"A":A,"w0":w0,"p":p},"metrics":{"rms":rms,"nrmse":nrmse},"inputs":args.inputs}
    with open(args.out_prefix+".json","w") as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta, indent=2))
if __name__=="__main__": main()
