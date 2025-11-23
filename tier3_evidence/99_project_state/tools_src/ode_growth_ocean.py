#!/usr/bin/env python3
"""
Phase-7: Ocean Dynamics (viscoelastic-GR 'hello world')
Integrates linear growth in N = ln(a):
  δ'' + [2 + (H'/H) + ν0] δ' - (3/2) μ0 Ω_m(a) δ = 0
with flat LCDM background H(a), returns fσ8(a) = [δ'/δ] * D(a) * σ8(0),
where D(a) = δ(a) / δ(a=1).

Usage examples:
  # minimal: build model column from ref CSV redshifts
  python3 tools/phase7_ocean/ode_growth_ocean.py \
    --ref-csv outputs/phase2/fs8_eval.csv \
    --out-csv outputs/phase7/fs8_eval_ocean.csv \
    --mu0 1.00 --nu0 0.05 --sigma8 0.81

  # then run Phase-6 on it:
  python3 tools/phase6_resonant_response_v5.py \
    --fs8-csv outputs/phase7/fs8_eval_ocean.csv \
    --z-col z --lcdm-col fs8_fit --model-col fs8_pred_ocean \
    --z-min 0.42 --z-max 0.75 \
    --max-gap 0.05 --eps 0.003 --floor-frac 0.0 \
    --clip-min 0.6 --clip-max 1.4
"""
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Background: flat LCDM ----------
def E_of_a(a, Om):
    """E(a) = H/H0 for flat LCDM with Ωm0=Om, ΩΛ0=1-Om."""
    return np.sqrt(Om * a**-3 + (1.0 - Om))

def Omega_m_of_a(a, Om):
    """Ωm(a) for flat LCDM."""
    Ea2 = E_of_a(a, Om)**2
    return (Om * a**-3) / Ea2

def dlnH_dN(a, Om):
    """(H'/H) = d ln H / dN for flat LCDM: = -(3/2) Ωm(a)."""
    return -1.5 * Omega_m_of_a(a, Om)

# ---------- ODE integrator in N = ln a ----------
def integrate_growth(mu0=1.0, nu0=0.0, Om=0.3, a_min=1e-3, a_max=1.0, n_steps=4000):
    """
    Integrate δ(N) from N_min=ln a_min to 0 (a=1).
    IC in matter era: δ ∝ a ⇒ At N0, set δ=e^{N0}, δ' = e^{N0}.
    Returns arrays: a_grid, D(a) (normalized), f(a) = d ln δ / d ln a.
    """
    N0 = math.log(a_min)
    N1 = math.log(a_max)
    N_grid = np.linspace(N0, N1, n_steps)
    a_grid = np.exp(N_grid)

    # State: y = [δ, δ'] with derivatives y' = [δ', δ'']
    # δ'' = -[2 + H'/H + ν0] δ' + (3/2) μ0 Ωm(a) δ
    def rhs(N, y):
        a = math.exp(N)
        dlnH = dlnH_dN(a, Om)
        Om_a = Omega_m_of_a(a, Om)
        delta, ddelta = y
        d2 = - (2.0 + dlnH + nu0) * ddelta + 1.5 * mu0 * Om_a * delta
        return np.array([ddelta, d2], dtype=float)

    # Initial conditions in matter-dom regime
    delta0 = math.exp(N0)
    ddelta0 = math.exp(N0)
    y = np.array([delta0, ddelta0], dtype=float)

    # 4th-order Runge-Kutta
    deltas = np.empty(n_steps)
    d_deltas = np.empty(n_steps)
    for i, N in enumerate(N_grid):
        deltas[i] = y[0]
        d_deltas[i] = y[1]
        if i == n_steps - 1:
            break
        h = N_grid[i+1] - N
        k1 = rhs(N, y)
        k2 = rhs(N + 0.5*h, y + 0.5*h*k1)
        k3 = rhs(N + 0.5*h, y + 0.5*h*k2)
        k4 = rhs(N + h,     y + h*k3)
        y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Normalize D(a=1) = 1
    D = deltas / deltas[-1]
    # f = d ln δ / d ln a = δ'/δ  (since N = ln a)
    f = d_deltas / deltas
    return a_grid, D, f

# ---------- Build fs8 predictions ----------
def fs8_from_D_f(a_grid, D, f, sigma8_0=0.81):
    """fs8(a) = f(a) * D(a) * σ8(0)."""
    return f * D * float(sigma8_0)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv", default="outputs/phase2/fs8_eval.csv",
                    help="CSV providing z (and optionally fs8_fit to carry through)")
    ap.add_argument("--out-csv", default="outputs/phase7/fs8_eval_ocean.csv")
    ap.add_argument("--H0", type=float, default=70.0)  # not used in ratios; here for completeness
    ap.add_argument("--Om", type=float, default=0.3)
    ap.add_argument("--mu0", type=float, default=1.0, help="ocean stiffness (μ0)")
    ap.add_argument("--nu0", type=float, default=0.0, help="ocean damping (ν0)")
    ap.add_argument("--sigma8", type=float, default=0.81, help="σ8(0)")
    ap.add_argument("--a-min", type=float, default=1e-3)
    ap.add_argument("--a-max", type=float, default=1.0)
    ap.add_argument("--n-steps", type=int, default=4000)
    args = ap.parse_args()

    Path("outputs/phase7").mkdir(parents=True, exist_ok=True)

    # Integrate growth
    a_grid, D, f = integrate_growth(mu0=args.mu0, nu0=args.nu0, Om=args.Om,
                                    a_min=args.a_min, a_max=args.a_max, n_steps=args.n_steps)
    fs8 = fs8_from_D_f(a_grid, D, f, sigma8_0=args.sigma8)
    z_grid = 1.0 / a_grid - 1.0

    # Load reference CSV for z & fs8_fit (if available)
    try:
        ref = pd.read_csv(args.ref_csv)
    except Exception:
        ref = pd.DataFrame()

    if "z" in ref.columns:
        # Interpolate model fs8 onto reference z
        z_ref = pd.to_numeric(ref["z"], errors="coerce").to_numpy()
        # restrict to our integration range
        valid = np.isfinite(z_ref) & (z_ref >= z_grid.min()) & (z_ref <= z_grid.max())
        fs8_interp = np.interp(z_ref[valid], z_grid[::-1], fs8[::-1])  # z decreases with a
        out = ref.copy()
        out.loc[valid, "fs8_pred_ocean"] = fs8_interp
        # Leave out-of-range rows NaN in fs8_pred_ocean so Phase-6 z-windowing can handle them
    else:
        out = pd.DataFrame({"z": z_grid, "fs8_pred_ocean": fs8})

    # Ensure we carry fs8_fit through if present (Phase-6 uses lcdm-col=fs8_fit)
    # If not present, we leave it absent; you can still compare using your Phase-6 settings if you point lcdm-col accordingly.
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(f"Params: mu0={args.mu0}, nu0={args.nu0}, Om={args.Om}, sigma8={args.sigma8}")
    print(f"a-range: [{args.a_min}, {args.a_max}]  n_steps={args.n_steps}")

if __name__ == "__main__":
    main()
