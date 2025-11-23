
#!/usr/bin/env python3
import argparse, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def eps_law(omega_hat, A, w0, p):
    return A * (1.0 - np.exp(- (omega_hat / w0)**p ))

def load_and_normalize(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # required columns: omega, omega0, epsilon
        need = {"omega","omega0","epsilon"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing required columns: {need - set(df.columns)}")
        df = df.copy()
        df["omega_hat"] = df["omega"] / df["omega0"]
        if "label" not in df.columns:
            df["label"] = os.path.basename(p)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["omega_hat","epsilon"])
        # clip epsilon to [0,1] just in case
        df["epsilon"] = df["epsilon"].clip(0.0, 1.0)
        frames.append(df[["omega_hat","epsilon","label"]])
    return pd.concat(frames, ignore_index=True)

def main():
    ap = argparse.ArgumentParser(description="Overlay universal ε(ω̂) collapse and fit law parameters")
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV files with omega, omega0, epsilon[,label]")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for plot and json")
    ap.add_argument("--bounds", default=None, help="Parameter bounds as JSON, e.g. '{\"A\":[0,2],\"w0\":[1e-6,10],\"p\":[0.1,10]}'")
    args = ap.parse_args()

    df = load_and_normalize(args.inputs)

    x = df["omega_hat"].values.astype(float)
    y = df["epsilon"].values.astype(float)

    p0 = [1.0, 1.0, 1.0]  # A, w0, p
    bounds = (-np.inf, np.inf)
    if args.bounds:
        b = json.loads(args.bounds)
        lower = [b.get("A",[0,-np.inf])[0], b.get("w0",[0,-np.inf])[0], b.get("p",[0,-np.inf])[0]]
        upper = [b.get("A",[np.inf,np.inf])[1] if len(b.get("A",[0,np.inf]))>1 else np.inf,
                 b.get("w0",[np.inf,np.inf])[1] if len(b.get("w0",[0,np.inf]))>1 else np.inf,
                 b.get("p",[np.inf,np.inf])[1] if len(b.get("p",[0,np.inf]))>1 else np.inf]
        bounds = (lower, upper)

    popt, pcov = curve_fit(eps_law, x, y, p0=p0, bounds=bounds, maxfev=200000)
    A, w0, p = map(float, popt)
    pred = eps_law(x, *popt)
    resid = y - pred
    rms = float(np.sqrt(np.mean(resid**2)))
    nrmse = float(rms / (y.max()-y.min() if y.max()>y.min() else 1.0))

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    # Plot: no explicit colors/styles, single axes
    fig, ax = plt.subplots()
    for label, sub in df.groupby("label"):
        ax.scatter(sub["omega_hat"], sub["epsilon"], s=12, label=label)
    # model curve on a clean grid
    gx = np.linspace(max(1e-8, x.min()), x.max(), 600)
    gy = eps_law(gx, *popt)
    ax.plot(gx, gy, linewidth=2)
    ax.set_xlabel("ω̂ = ω / ω0")
    ax.set_ylabel("ε (energy localization)")
    ax.set_title("Universal ε(ω̂) collapse with fit")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.out_prefix + ".png", dpi=200)
    plt.close(fig)

    meta = {
        "fit_params": {"A": A, "w0": w0, "p": p},
        "metrics": {"rms": rms, "nrmse": nrmse},
        "inputs": args.inputs,
        "notes": "Model: ε(ω̂)=A[1-exp(-(ω̂/w0)^p)]"
    }
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
