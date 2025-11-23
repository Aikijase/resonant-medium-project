#!/usr/bin/env python3
# viscous_dm_growth.py
# Linear matter growth with an effective dark-matter viscosity (and optional tiny sound speed).
# Author: ChatGPT (Jason's teammate). License: MIT.
#
# Usage example:
#   python3 viscous_dm_growth.py --nu0 0.5 --cs2 0.0 --kmin 0.05 --kmax 3.0 --nk 30 \
#       --a-min 0.001 --n-a 800 --Om0 0.3 --h 0.7 --s-nu 0.0 --out-root outputs/visc_dm_demo
#
# Exact k list:
#   python3 viscous_dm_growth.py --nu0 0.5 --cs2 0.0 --Om0 0.3 --h 0.7 \
#       --k-list "0.05,0.10,0.20,0.30" --a-min 1e-3 --n-a 800 \
#       --out-root outputs/joint_visc_dm
#
# Outputs:
#   <out-root>_growth_grid.csv      : a-grid with LCDM D(a) and D_visc(a,k) for all k
#   <out-root>_summary_z0.csv       : per-k summary at a=1 (z=0) with suppression D_visc/D_LCDM
#   <out-root>_growth_curves.png    : plot of D(a)/a for selected k
#   <out-root>_suppression_vs_k.png : plot of suppression at z=0 vs k
#
# Notes:
# - Units: integrate with k in [h/Mpc], internally convert to [1/Mpc] using h.
#          H is converted to [1/Gyr] for numerical stability. ν must be in [Mpc^2/Gyr].
# - Viscous term is + (ν k^2 / a^2) D' in the N=ln a equation (see derivation).
# - Set cs2=0 for pure viscosity. Tiny cs2 adds Jeans-like pressure: + (cs2 k^2 / a^2 H^2) D.
#
# Growth ODE in N = ln a (sub-horizon, Newtonian gauge):
#   D'' + [2 + d(ln H)/dN + ν(a) k^2/(a^2 H)] D'
#       + [ (c_s^2 k^2)/(a^2 H^2) - (3/2) Ω_m(a) ] D = 0
#
# Integrator: RK4 in N.

import argparse, os, math, csv, json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ---------- Physical constants ----------
MPC_IN_KM = 3.0856775814913673e19  # km
SEC_PER_YEAR = 3.15576e7
SEC_PER_GYR  = SEC_PER_YEAR * 1.0e9

def H0_to_per_Gyr(H0_km_s_Mpc: float) -> float:
    """Convert H0 from (km/s)/Mpc to 1/Gyr."""
    H0_per_s = H0_km_s_Mpc / MPC_IN_KM
    return H0_per_s * SEC_PER_GYR

# ---------- Background (flat ΛCDM) ----------
def E_of_a(a, Om0, Ode0):
    # flat LCDM: E(a)^2 = Om0 a^{-3} + Ode0
    return math.sqrt(Om0 * a**(-3) + Ode0)

def H_of_a_per_Gyr(a, H0_per_Gyr, Om0, Ode0):
    return H0_per_Gyr * E_of_a(a, Om0, Ode0)

def Omega_m_of_a(a, Om0, Ode0):
    return (Om0 * a**(-3)) / (Om0 * a**(-3) + Ode0)

def dlnH_dN(a, Om0, Ode0):
    # d ln H / dN for flat LCDM: = - (3/2) * Omega_m(a)
    return -1.5 * Omega_m_of_a(a, Om0, Ode0)

# ---------- ν(a) and c_s^2(a) models ----------
def nu_of_a(a, nu0, s_nu):
    # Simple power law: nu(a) = nu0 * a^{s_nu}
    return nu0 * (a ** s_nu)

def cs2_of_a(a, cs2_const):
    # constant (can be 0.0). Dimensionless (c=1 units).
    return cs2_const

# ---------- RK4 Solver for D(N) ----------
def solve_growth_for_k(k_h_Mpc, a_grid, H0_per_Gyr, Om0, Ode0, h_param, nu0, s_nu, cs2_const):
    """
    Solve growth ODE for a given k (in h/Mpc). Returns D(a) normalized to D(a=1)=1.
    Signature used by external runners; keep as-is.
    """
    # Convert k from [h/Mpc] to [1/Mpc]
    k = (k_h_Mpc * h_param)  # 1/Mpc

    N = np.log(a_grid)
    D  = np.zeros_like(N)
    Dp = np.zeros_like(N)

    # Initial conditions: deep in matter era, growing mode ~ a → D' ≈ 1, D ≈ a
    D[0]  = a_grid[0]
    Dp[0] = 1.0

    def A_of(a):
        H = H_of_a_per_Gyr(a, H0_per_Gyr, Om0, Ode0)
        return 2.0 + dlnH_dN(a, Om0, Ode0) + (nu_of_a(a, nu0, s_nu) * (k**2)) / (a*a * H)

    def B_of(a):
        H = H_of_a_per_Gyr(a, H0_per_Gyr, Om0, Ode0)
        return (cs2_of_a(a, cs2_const) * (k**2)) / (a*a * H*H) - 1.5 * Omega_m_of_a(a, Om0, Ode0)

    for i in range(len(N)-1):
        a  = a_grid[i]
        hN = N[i+1] - N[i]

        # y = [D, Dp], y' = [Dp, -A Dp - B D]
        def deriv(a_local, D_local, Dp_local):
            A = A_of(a_local)
            B = B_of(a_local)
            Dpp = -A * Dp_local - B * D_local
            return Dp_local, Dpp

        # RK4 in N, with a = e^N entering A,B
        k1_Dp, k1_Dpp = deriv(a, D[i], Dp[i])

        a2 = np.exp(N[i] + 0.5*hN)
        k2_Dp, k2_Dpp = deriv(a2, D[i] + 0.5*hN*k1_Dp, Dp[i] + 0.5*hN*k1_Dpp)

        k3_Dp, k3_Dpp = deriv(a2, D[i] + 0.5*hN*k2_Dp, Dp[i] + 0.5*hN*k2_Dpp)

        a4 = np.exp(N[i] + hN)
        k4_Dp, k4_Dpp = deriv(a4, D[i] + hN*k3_Dp, Dp[i] + hN*k3_Dpp)

        D[i+1]  = D[i]  + (hN/6.0)*(k1_Dp + 2*k2_Dp + 2*k3_Dp + k4_Dp)
        Dp[i+1] = Dp[i] + (hN/6.0)*(k1_Dpp+ 2*k2_Dpp+ 2*k3_Dpp+ k4_Dpp)

    # Normalize to D(a=1)=1
    if D[-1] == 0.0 or not np.isfinite(D[-1]):
        return D
    return D / D[-1]

def solve_LCDM_growth(a_grid, H0_per_Gyr, Om0, Ode0):
    """
    Solve standard LCDM scale-independent growth (ν=0, c_s^2=0, k→0).
    Return D(a) normalized to D(1)=1.
    """
    N = np.log(a_grid)
    D  = np.zeros_like(N)
    Dp = np.zeros_like(N)

    D[0]  = a_grid[0]
    Dp[0] = 1.0

    def A_of(a):
        return 2.0 + dlnH_dN(a, Om0, Ode0)

    def B_of(a):
        return -1.5 * Omega_m_of_a(a, Om0, Ode0)

    for i in range(len(N)-1):
        a  = a_grid[i]
        hN = N[i+1] - N[i]

        def deriv(a_local, D_local, Dp_local):
            A = A_of(a_local)
            B = B_of(a_local)
            Dpp = -A * Dp_local - B * D_local
            return Dp_local, Dpp

        k1_Dp, k1_Dpp = deriv(a, D[i], Dp[i])

        a2 = np.exp(N[i] + 0.5*hN)
        k2_Dp, k2_Dpp = deriv(a2, D[i] + 0.5*hN*k1_Dp, Dp[i] + 0.5*hN*k1_Dpp)

        k3_Dp, k3_Dpp = deriv(a2, D[i] + 0.5*hN*k2_Dp, Dp[i] + 0.5*hN*k2_Dpp)

        a4 = np.exp(N[i] + hN)
        k4_Dp, k4_Dpp = deriv(a4, D[i] + hN*k3_Dp, Dp[i] + hN*k3_Dpp)

        D[i+1]  = D[i]  + (hN/6.0)*(k1_Dp + 2*k2_Dp + 2*k3_Dp + k4_Dp)
        Dp[i+1] = Dp[i] + (hN/6.0)*(k1_Dpp+ 2*k2_Dpp+ 2*k3_Dpp+ k4_Dpp)

    return D / D[-1]

def main():
    ap = argparse.ArgumentParser(description="Viscous-DM linear growth (with optional tiny sound speed).")
    ap.add_argument("--Om0", type=float, default=0.3, help="Omega_m,0 today (default 0.3)")
    ap.add_argument("--h",   type=float, default=0.7, help="little h (H0/100), default 0.7")
    ap.add_argument("--nu0", type=float, default=0.5, help="viscosity at a=1, in Mpc^2/Gyr (default 0.5)")
    ap.add_argument("--s-nu", type=float, default=0.0, help="power-law index for nu(a)=nu0*a^{s_nu} (default 0)")
    ap.add_argument("--cs2", type=float, default=0.0, help="constant sound speed^2 (dimensionless), default 0")
    ap.add_argument("--kmin", type=float, default=0.05, help="min k in h/Mpc (default 0.05)")
    ap.add_argument("--kmax", type=float, default=3.0, help="max k in h/Mpc (default 3.0)")
    ap.add_argument("--nk",   type=int,   default=30, help="number of k samples (default 30)")
    ap.add_argument("--k-list", type=str, default=None,
                    help="Comma list of k in h/Mpc (overrides kmin/kmax/nk), e.g. '0.05,0.10,0.20,0.30'")
    ap.add_argument("--a-min", type=float, default=1e-3, help="start a (default 1e-3)")
    ap.add_argument("--n-a",   type=int,   default=800, help="number of scale factor steps (default 800)")
    ap.add_argument("--out-root", type=str, default="outputs/visc_dm_demo", help="output root path (no extension)")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    args = ap.parse_args()

    Om0 = args.Om0
    h   = args.h
    Ode0 = 1.0 - Om0  # flat LCDM

    # Grids
    a_grid = np.linspace(args.a_min, 1.0, args.n_a)
    if args.k_list:
        k_grid = np.array([float(s) for s in args.k_list.split(",")], dtype=float)
    else:
        k_grid = np.linspace(args.kmin, args.kmax, args.nk)

    # Background H0 in 1/Gyr
    H0_km_s_Mpc = 100.0 * h  # km/s/Mpc
    H0_per_Gyr = H0_to_per_Gyr(H0_km_s_Mpc)

    # Solve LCDM baseline
    D_LCDM = solve_LCDM_growth(a_grid, H0_per_Gyr, Om0, Ode0)

    # Solve viscous growth for all k
    D_visc_all = []
    for k in k_grid:
        Dk = solve_growth_for_k(k, a_grid, H0_per_Gyr, Om0, Ode0, h, args.nu0, args.s_nu, args.cs2)
        D_visc_all.append(Dk)
    D_visc_all = np.array(D_visc_all)  # shape (nk, n_a)

    # Prepare outputs
    out_root = args.out_root
    out_dir = os.path.dirname(out_root)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1) Growth grid CSV
    grid_csv = out_root + "_growth_grid.csv"
    with open(grid_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["a", "D_LCDM"] + [f"D_visc_k={k_grid[idx]:.6f}" for idx in range(len(k_grid))]
        w.writerow(header)
        for i, a in enumerate(a_grid):
            row = [f"{a:.8e}", f"{D_LCDM[i]:.8e}"] + [f"{D_visc_all[j, i]:.8e}" for j in range(len(k_grid))]
            w.writerow(row)

    # 2) Summary at z=0 (a=1)
    summary_csv = out_root + "_summary_z0.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k_hMpc", "D_LCDM(a=1)", "D_visc(a=1)", "suppression=D_visc/D_LCDM"])
        for j, k in enumerate(k_grid):
            Dv = float(D_visc_all[j, -1])
            w.writerow([f"{k:.6f}", f"{D_LCDM[-1]:.8e}", f"{Dv:.8e}", f"{(Dv / D_LCDM[-1]):.8e}"])

    # 3) Plots (each chart on its own figure)
    if not args.no_plots:
        # Plot 1: D(a)/a for LCDM and a few k's
        plt.figure()
        ratio_LCDM = D_LCDM / a_grid
        plt.plot(a_grid, ratio_LCDM, label="LCDM (D/a)")
        # pick 3 representative ks (low, mid, high)
        ks_pick_idx = [0, len(k_grid)//2, len(k_grid)-1]
        for idx in ks_pick_idx:
            ratio = D_visc_all[idx, :] / a_grid
            plt.plot(a_grid, ratio, label=f"visc k={k_grid[idx]:.3f} h/Mpc")
        plt.xlabel("scale factor a")
        plt.ylabel("D(a)/a")
        plt.legend(loc="best")
        plt.title("Growth ratio D(a)/a (LCDM vs viscous DM)")
        fig1 = out_root + "_growth_curves.png"
        plt.tight_layout()
        plt.savefig(fig1, dpi=140)
        plt.close()

        # Plot 2: suppression at z=0 vs k
        plt.figure()
        suppression = np.array([D_visc_all[j, -1] / D_LCDM[-1] for j in range(len(k_grid))])
        plt.plot(k_grid, suppression, marker="o")
        plt.xlabel("k [h/Mpc]")
        plt.ylabel("D_visc(a=1)/D_LCDM(a=1)")
        plt.title("Suppression at z=0 from viscous DM")
        fig2 = out_root + "_suppression_vs_k.png"
        plt.tight_layout()
        plt.savefig(fig2, dpi=140)
        plt.close()
    else:
        fig1 = fig2 = None

    # 4) Metadata JSON
    meta = {
        "args": vars(args),
        "H0_km_s_Mpc": H0_km_s_Mpc,
        "H0_per_Gyr": H0_per_Gyr,
        "k_grid": list(map(float, k_grid)),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(out_root + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    print("  ", grid_csv)
    print("  ", summary_csv)
    if fig1: print("  ", fig1)
    if fig2: print("  ", fig2)
    print("  ", out_root + "_meta.json")

if __name__ == "__main__":
    main()
