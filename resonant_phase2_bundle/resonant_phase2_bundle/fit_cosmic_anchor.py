
#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def eps_law(omega_hat, A, w0, p):
    return A * (1.0 - np.exp(- (omega_hat / w0)**p ))

def default_horizon_frequency(z, H0=70.0, Om=0.3, Ol=0.7):
    # Toy mapping: ω ∝ H(z). Use ω̂ = H(z)/H0 by default.
    # Replace with your membrane-driven mapping if preferred.
    def E(z):
        return np.sqrt(Om*(1+z)**3 + Ol)
    return E(z)  # dimensionless ω̂ if referenced to H0

def main():
    ap = argparse.ArgumentParser(description="Fit ε(ω̂) to a cosmic anchor mapping using ρ_BH(z) or H(z)")
    ap.add_argument("--bh-density", help="CSV with columns z, rho_BH (optional for overlay)")
    ap.add_argument("--out", required=True, help="Output JSON (and PNG) prefix")
    ap.add_argument("--bounds", default=None, help="Bounds JSON for (A,w0,p) like '{\"A\":[0,2],\"w0\":[1e-3,10],\"p\":[0.1,10]}'")
    args = ap.parse_args()

    # Build a synthetic epsilon(z) proxy from ρ_BH(z) if provided (monotone rescale).
    zgrid = np.linspace(0, 6, 200)
    omega_hat = default_horizon_frequency(zgrid)

    # If BH density provided, monotonic map to epsilon target in [0,1].
    y_target = None
    if args.bh_density:
        df = pd.read_csv(args.bh_density)
        df = df.dropna(subset=["z"]).sort_values("z")
        # normalize rho_BH → [0,1]
        if "rho_BH" in df.columns:
            r = df["rho_BH"].values.astype(float)
            r = (r - r.min())/(r.max()-r.min() if r.max()>r.min() else 1.0)
            # interpolate onto zgrid as target ε(z)
            y_target = np.interp(zgrid, df["z"].values, r)
        else:
            print("Warning: rho_BH column not found; proceeding without target ε(z)")

    # Fit eps_law(omega_hat) to the target if present; otherwise just report a nominal fit to a synthetic trend.
    if y_target is None:
        # simple synthetic increasing target to demonstrate the fit
        y_target = 1.0 - np.exp(-omega_hat)

    p0 = [1.0, 1.0, 1.0]
    bounds = (-np.inf, np.inf)
    if args.bounds:
        b = json.loads(args.bounds)
        lower = [b.get("A",[0,-np.inf])[0], b.get("w0",[0,-np.inf])[0], b.get("p",[0,-np.inf])[0]]
        upper = [b.get("A",[np.inf,np.inf])[1] if len(b.get("A",[0,np.inf]))>1 else np.inf,
                 b.get("w0",[np.inf,np.inf])[1] if len(b.get("w0",[0,np.inf]))>1 else np.inf,
                 b.get("p",[np.inf,np.inf])[1] if len(b.get("p",[0,np.inf]))>1 else np.inf]
        bounds = (lower, upper)

    popt, pcov = curve_fit(eps_law, omega_hat, y_target, p0=p0, bounds=bounds, maxfev=200000)
    A, w0, p = map(float, popt)
    pred = eps_law(omega_hat, *popt)
    resid = y_target - pred
    rms = float(np.sqrt(np.mean(resid**2)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Plot (no explicit colors/styles)
    fig, ax = plt.subplots()
    ax.plot(omega_hat, y_target, linewidth=2)
    ax.plot(omega_hat, pred, linewidth=2)
    ax.set_xlabel("ω̂(z) (toy: H(z)/H0)")
    ax.set_ylabel("ε(z)")
    ax.set_title("Cosmic anchor ε fit (toy mapping)")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=200)
    print(json.dumps({
        "fit_params": {"A": A, "w0": w0, "p": p},
        "metrics": {"rms": rms},
        "notes": "Replace default_horizon_frequency with your membrane/anchor mapping; supply rho_BH(z) to guide ε target."
    }, indent=2))

    with open(args.out + ".json", "w") as f:
        json.dump({
            "fit_params": {"A": A, "w0": w0, "p": p},
            "metrics": {"rms": rms},
            "mapping": "omega_hat(z)=H(z)/H0 (default, replace if needed)",
        }, f, indent=2)

if __name__ == "__main__":
    main()
