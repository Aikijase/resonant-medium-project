#!/usr/bin/env python3
"""
Phase-2 Dwarf Subhalo Count Estimator
Integrates the Press–Schechter HMF to estimate the number of subhalos
in a Milky-Way–like host (simple scaling). This is a back-of-envelope tool
for Missing Satellites / TBTF checks.

Usage:
  python3 tools/phase2_dwarfcount.py --tau 0.05 --omega 0.144 --Mhost 1e12 \
    --Mmin 1e9 --Mmax 1e10 --Rvir 250.0

Outputs a small JSON with counts for LCDM and Resonant models.
"""

import argparse, json
import numpy as np

# Cosmology (consistent with sweep script)
h = 0.7
Omega_m = 0.3
n_s = 0.965
rho_crit = 2.775e11  # Msun/(Mpc/h)^3
rho_m = Omega_m * rho_crit

def bbks_transfer(k, Omega_m=Omega_m, h=h):
    Gamma = Omega_m * h
    q = k / Gamma
    T = np.log(1 + 2.34*q) / (2.34*q)
    T *= (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4) ** (-0.25)
    return T

def P_lcdm_linear(k):
    T = bbks_transfer(k)
    return (k**n_s) * (T**2)

def resonant_P(P_lcdm, k, tau, omega):
    return P_lcdm * np.exp(-tau * k**2) * (np.cos(omega * k)**2)

def W_tophat(x):
    W = np.ones_like(x)
    small = x < 1e-4
    xs = x[small]
    if xs.size > 0:
        W[small] = 1 - xs**2/10.0
    big = ~small
    xb = x[big]
    W[big] = 3*(np.sin(xb) - xb*np.cos(xb))/ (xb**3 + 1e-300)
    return W

def sigma_M(Pk, k, M):
    R = (3*M/(4*np.pi*rho_m))**(1/3)
    x = k * R
    W = W_tophat(x)
    integrand = (k**2) * Pk * (W**2)
    sig2 = np.trapz(integrand, k) / (2*np.pi**2)
    return float(np.sqrt(max(sig2, 1e-30)))

def hmf_ps(M, sigma):
    delta_c = 1.686
    lnM = np.log(M)
    ln_sig_inv = np.log(1.0/ sigma)
    dln_sig_inv_dlnM = np.gradient(ln_sig_inv, lnM)
    nu = delta_c / sigma
    f = np.sqrt(2/np.pi) * nu * np.exp(-0.5*nu**2)
    dn = (rho_m / M) * f * np.abs(dln_sig_inv_dlnM)
    dn = np.where(np.isfinite(dn) & (dn>0), dn, np.nan)
    return dn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--omega", type=float, required=True)
    ap.add_argument("--Mhost", type=float, default=1e12, help="Host halo mass Msun/h")
    ap.add_argument("--Mmin", type=float, default=1e9,  help="Min dwarf mass Msun/h")
    ap.add_argument("--Mmax", type=float, default=1e10, help="Max dwarf mass Msun/h")
    ap.add_argument("--Rvir", type=float, default=250.0, help="Host virial radius [kpc/h], rough volume scaling")
    args = ap.parse_args()

    # k-grid and spectra
    k = np.logspace(-3, 1.4, 1600)
    P_lin = P_lcdm_linear(k)
    P_res = resonant_P(P_lin, k, args.tau, args.omega)

    # Build HMF on a broad mass grid
    M_grid = np.logspace(8, 12, 400)
    sigma_LCDM = np.array([sigma_M(P_lin, k, M) for M in M_grid])
    sigma_RES  = np.array([sigma_M(P_res, k, M) for M in M_grid])
    dn_LCDM = hmf_ps(M_grid, sigma_LCDM)
    dn_RES  = hmf_ps(M_grid, sigma_RES)

    # Integrate number density over [Mmin, Mmax]
    # dn/dlnM -> n = ∫ (dn/dlnM) dlnM
    lnM = np.log(M_grid)
    mask = (M_grid >= args.Mmin) & (M_grid <= args.Mmax)
    n_LCDM = float(np.trapz(dn_LCDM[mask], lnM[mask]))
    n_RES  = float(np.trapz(dn_RES[mask], lnM[mask]))

    # Convert number density to expected count inside host volume.
    # Crude proxy: use host's virial sphere volume to scale.
    R_Mpch = args.Rvir / 1000.0  # kpc/h -> Mpc/h
    V_host = (4/3)*np.pi*(R_Mpch**3)  # (Mpc/h)^3

    N_LCDM = n_LCDM * V_host
    N_RES  = n_RES  * V_host

    out = {
        "tau": args.tau, "omega": args.omega,
        "Mhost_Msun_h": args.Mhost,
        "M_range_Msun_h": [args.Mmin, args.Mmax],
        "Rvir_kpc_h": args.Rvir,
        "ndensity_LCDM_Mpch3": n_LCDM,
        "ndensity_RES_Mpch3":  n_RES,
        "N_expected_LCDM": N_LCDM,
        "N_expected_RES":  N_RES,
        "suppression_ratio": (N_RES / N_LCDM) if (N_LCDM>0) else None
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
