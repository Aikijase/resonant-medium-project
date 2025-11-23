#!/usr/bin/env python3
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def eps_law(oh, A, w0, p):
    oh = np.clip(oh, 1e-12, np.inf)   # avoid 0^p
    w0 = max(float(w0), 1e-12)
    return A * (1.0 - np.exp(- (oh / w0)**p))

def Ez(z, Om, Ol):
    return np.sqrt(Om*(1.0+z)**3 + Ol)

def moving_average(y, k):
    if k <= 1: return y
    k = int(k)
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, np.ones(k)/k, mode="valid")[:len(y)]

def main():
    ap = argparse.ArgumentParser(description="Fit ε(ω̂) to a cosmic anchor mapping")
    ap.add_argument("--bh-density", help="CSV with columns z,rho_BH (optional)")
    ap.add_argument("--out", required=True, help="Output prefix (PNG/JSON/CSV)")
    ap.add_argument("--map", choices=["Hz","invH"], default="Hz", help="ω̂ mapping: Hz = H(z)/H0, invH = 1/E(z)")
    ap.add_argument("--Om", type=float, default=0.3)
    ap.add_argument("--Ol", type=float, default=0.7)
    ap.add_argument("--zmax", type=float, default=6.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window for rho_BH")
    ap.add_argument("--bounds", default="", help='JSON like {"A":[0,2],"w0":[1e-3,10],"p":[0.1,12]}')
    args = ap.parse_args()

    # Build ω̂(z)
    z = np.linspace(0.0, args.zmax, args.n)
    E = Ez(z, args.Om, args.Ol)
    oh = E if args.map == "Hz" else 1.0 / np.clip(E, 1e-12, np.inf)

    # Target ε(z) from rho_BH(z) if provided, else synthetic
    y = None
    if args.bh_density and os.path.exists(args.bh_density):
        df = pd.read_csv(args.bh_density).dropna(subset=["z"]).sort_values("z")
        if "rho_BH" in df:
            r = df["rho_BH"].astype(float).to_numpy()
            if args.smooth > 1:
                r = moving_average(r, args.smooth)
            rmin, rmax = float(np.min(r)), float(np.max(r))
            denom = (rmax - rmin) if rmax > rmin else 1.0
            r = (r - rmin) / denom
            y = np.interp(z, df["z"].astype(float).to_numpy(), r)

    if y is None:
        y = 1.0 - np.exp(-oh)

    # Clean / clip
    y = np.clip(y, 0.0, 1.0)
    mask = np.isfinite(oh) & np.isfinite(y)
    oh_m, y_m = oh[mask], y[mask]

    # Bounds
    lb = np.array([0.0, 1e-3, 0.1])
    ub = np.array([2.0, 10.0, 12.0])
    if args.bounds:
        b = json.loads(args.bounds)
        for i, k in enumerate(["A","w0","p"]):
            if k in b and len(b[k]) == 2:
                lb[i], ub[i] = float(b[k][0]), float(b[k][1])

    p0 = [1.0, 1.0, 1.0]
    popt, _ = curve_fit(eps_law, oh_m, y_m, p0=p0, bounds=(lb, ub), maxfev=200000)
    A, w0, p = map(float, popt)
    pred = eps_law(oh, *popt)
    resid = y - pred
    rms = float(np.sqrt(np.mean(resid[mask]**2)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Plot (headless)
    plt.plot(oh, y); plt.plot(oh, pred)
    plt.xlabel(f"ω̂(z) ({args.map})"); plt.ylabel("ε(z)"); plt.title("Cosmic anchor ε fit")
    plt.tight_layout(); plt.savefig(args.out + ".png", dpi=200); plt.close()

    # Save JSON + residuals
    with open(args.out + ".json","w") as f:
        json.dump({
            "fit_params":{"A":A,"w0":w0,"p":p},
            "metrics":{"rms":rms},
            "mapping":args.map,
            "z_range":[float(z.min()), float(z.max())],
            "n_points": int(len(z))
        }, f, indent=2)

    pd.DataFrame({
        "z": z, "omega_hat": oh, "epsilon_target": y,
        "epsilon_fit": pred, "residual": resid
    }).to_csv(args.out + "_residuals.csv", index=False)

    print(json.dumps({"A":A,"w0":w0,"p":p,"rms":rms,"map":args.map}, indent=2))

if __name__ == "__main__":
    main()
