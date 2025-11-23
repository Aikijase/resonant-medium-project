#!/usr/bin/env python3
import json, argparse
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

def law(oh,A,w0,p):
    oh = np.clip(oh, 1e-12, np.inf)
    w0 = max(float(w0), 1e-12)
    return A*(1 - np.exp(- (oh/w0)**p))

def load(path):
    df = pd.read_csv(path)
    if "omega_hat" in df.columns:
        oh = df["omega_hat"].to_numpy(float)
    else:
        oh = (df["omega"]/df["omega0"]).to_numpy(float)
    y  = df["epsilon"].to_numpy(float)
    m = np.isfinite(oh) & np.isfinite(y)
    return oh[m], np.clip(y[m], 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bec",   default="data/bec/steinhauer2016_ridgeA.csv")
    ap.add_argument("--abh",   default="data/abh_plate_norm.csv")
    ap.add_argument("--dusty", default="data/dusty_plasma_norm.csv")
    ap.add_argument("--out",   default="outputs/sweep_dusty_sb.json")
    ap.add_argument("--write", default="data/dusty_plasma_norm_sb.csv")
    args = ap.parse_args()

    oh_bec, y_bec = load(args.bec)
    oh_abh, y_abh = load(args.abh)
    oh_dpl0, y_dpl0 = load(args.dusty)

    best = None
    for s in np.arange(0.90, 1.111, 0.01):       # gentle scale range
        for b in np.arange(-0.05, 0.051, 0.005): # tiny offset range
            oh_dpl = oh_dpl0*s + b
            X = np.concatenate([oh_bec, oh_abh, oh_dpl])
            Y = np.concatenate([y_bec,  y_abh,  y_dpl0])
            L = (["BEC"]*len(oh_bec)) + (["ABH"]*len(oh_abh)) + (["DPL"]*len(oh_dpl))
            counts = {"BEC":len(oh_bec),"ABH":len(oh_abh),"DPL":len(oh_dpl)}
            sigma  = np.array([np.sqrt(counts[l]) for l in L], float)
            lb, ub = [0.0,1e-3,0.1], [1.2,10.0,8.0]
            try:
                (A,w0,p),_ = curve_fit(law, X, Y, p0=[1,1,1],
                                       sigma=sigma, bounds=(lb,ub), maxfev=200000)
                rms = float(np.sqrt(np.mean((Y - law(X,A,w0,p))**2)))
                if best is None or rms < best["rms"]:
                    best = {"s":float(s),"b":float(b),"A":float(A),
                            "w0":float(w0),"p":float(p),"rms":rms}
            except Exception:
                pass

    if best is None:
        raise SystemExit("Sweep failed to find a fit.")

    # write best result and transformed dusty file
    with open(args.out, "w") as f:
        json.dump(best, f, indent=2)

    oh_dpl = (oh_dpl0*best["s"] + best["b"])
    pd.DataFrame({
        "omega_hat": oh_dpl,
        "epsilon":   y_dpl0,
        "label":     "Dusty plasma (scaled/shifted)"
    }).to_csv(args.write, index=False)

    print(json.dumps(best, indent=2))
    print(f"Wrote {args.write}")

if __name__ == "__main__":
    main()
