#!/usr/bin/env python3
"""
Phase-7D: Ocean Dynamics with single-pole memory (tau, kappa)

States (in N = ln a):
  s'   = (delta' - s)/tau
  delta'' = -[2 + H'/H + nu0] * delta' + (3/2) * mu0 * Omega_m(a) * delta - kappa * s

Limits:
  - kappa = 0 (or tau -> 0) recovers the viscous-GR model from ode_growth_ocean.py
  - mu0 = 1, nu0 = 0, kappa = 0 gives LCDM

Outputs a CSV aligned to --ref-csv (by z) with column: fs8_pred_ocean_tau
"""

import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Background: flat LCDM ----------
def E_of_a(a, Om):
    return np.sqrt(Om * a**-3 + (1.0 - Om))

def Omega_m_of_a(a, Om):
    Ea2 = E_of_a(a, Om)**2
    return (Om * a**-3) / Ea2

def dlnH_dN(a, Om):
    # d(ln H)/dN = -(3/2) * Omega_m(a) for flat LCDM
    return -1.5 * Omega_m_of_a(a, Om)

# ---------- Integrator with memory ----------
def integrate_growth_tau(mu0=1.0, nu0=0.0, tau=0.1, kappa=0.02,
                         Om=0.3, a_min=1e-3, a_max=1.0, n_steps=4000):
    """
    Integrate [s, delta, ddelta] in N = ln a.
    IC in early matter era: delta ~ a, so at N0 set delta=e^{N0}, ddelta=e^{N0}.
    Set s(N0) = ddelta(N0) so the memory state starts on the attractor.
    Returns: a_grid, D(a), f(a).
    """
    N0 = math.log(a_min)
    N1 = math.log(a_max)
    N_grid = np.linspace(N0, N1, n_steps)
    a_grid = np.exp(N_grid)

    def rhs(N, y):
        # y = [s, delta, ddelta]
        s, delta, ddelta = y
        a = math.exp(N)
        dlnH = dlnH_dN(a, Om)
        Om_a = Omega_m_of_a(a, Om)

        # memory: s' = (delta' - s)/tau
        ds = (ddelta - s) / tau

        # main: delta'' = -[2 + H'/H + nu0] delta' + (3/2) mu0 Omega_m delta - kappa s
        d2 = - (2.0 + dlnH + nu0) * ddelta + 1.5 * mu0 * Om_a * delta - kappa * s

        return np.array([ds, ddelta, d2], dtype=float)

    # initial conditions
    delta0  = math.exp(N0)
    ddelta0 = math.exp(N0)
    s0      = ddelta0  # start memory state aligned with delta'
    y = np.array([s0, delta0, ddelta0], dtype=float)

    deltas  = np.empty(n_steps)
    d_deltas= np.empty(n_steps)

    # RK4 integration
    for i, N in enumerate(N_grid):
        deltas[i]   = y[1]
        d_deltas[i] = y[2]
        if i == n_steps - 1:
            break
        h = N_grid[i+1] - N
        k1 = rhs(N, y)
        k2 = rhs(N + 0.5*h, y + 0.5*h*k1)
        k3 = rhs(N + 0.5*h, y + 0.5*h*k2)
        k4 = rhs(N + h,     y + h*k3)
        y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Normalize growth to D(a=1)=1
    D = deltas / deltas[-1]
    # f = d ln delta / d ln a = delta' / delta (since N=ln a)
    f = d_deltas / deltas
    return a_grid, D, f

def fs8_from_D_f(a_grid, D, f, sigma8_0=0.81):
    return f * D * float(sigma8_0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv", default="outputs/phase2/fs8_eval.csv",
                    help="CSV providing z (and optionally fs8_fit to carry through)")
    ap.add_argument("--out-csv", default="outputs/phase7/fs8_eval_ocean_tau.csv")

    ap.add_argument("--Om", type=float, default=0.3)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--nu0", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.1, help="memory time (e-folds)")
    ap.add_argument("--kappa", type=float, default=0.02, help="memory strength")
    ap.add_argument("--sigma8", type=float, default=0.81)

    ap.add_argument("--a-min", type=float, default=1e-3)
    ap.add_argument("--a-max", type=float, default=1.0)
    ap.add_argument("--n-steps", type=int, default=4000)
    args = ap.parse_args()

    Path("outputs/phase7").mkdir(parents=True, exist_ok=True)

    a_grid, D, f = integrate_growth_tau(mu0=args.mu0, nu0=args.nu0, tau=args.tau, kappa=args.kappa,
                                        Om=args.Om, a_min=args.a_min, a_max=args.a_max, n_steps=args.n_steps)
    fs8 = fs8_from_D_f(a_grid, D, f, sigma8_0=args.sigma8)
    z_grid = 1.0/a_grid - 1.0

    # Map onto ref z
    try:
        ref = pd.read_csv(args.ref_csv)
    except Exception:
        ref = pd.DataFrame()

    if "z" in ref.columns:
        z_ref = pd.to_numeric(ref["z"], errors="coerce").to_numpy()
        valid = np.isfinite(z_ref) & (z_ref >= z_grid.min()) & (z_ref <= z_grid.max())
        fs8_interp = np.interp(z_ref[valid], z_grid[::-1], fs8[::-1])  # z decreases with a
        out = ref.copy()
        out.loc[valid, "fs8_pred_ocean_tau"] = fs8_interp
    else:
        out = pd.DataFrame({"z": z_grid, "fs8_pred_ocean_tau": fs8})

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(f"Params: mu0={args.mu0}, nu0={args.nu0}, tau={args.tau}, kappa={args.kappa}, Om={args.Om}, sigma8={args.sigma8}")
    print(f"a-range: [{args.a_min}, {args.a_max}]  n_steps={args.n_steps}")

if __name__ == "__main__":
    main()
