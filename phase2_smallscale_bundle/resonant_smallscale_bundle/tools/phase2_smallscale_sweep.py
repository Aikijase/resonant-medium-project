#!/usr/bin/env python3
"""
Phase-2 Small-Scale Sweep (ΛCDM vs Resonant)
Generates P(k), σ(M), Halo Mass Function, and a (τ, ω) mini-sweep tuned for dwarf-scale suppression.
Outputs are saved to outputs/phase2_smallscale/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
OUTDIR = "outputs/phase2_smallscale"
os.makedirs(OUTDIR, exist_ok=True)

# Cosmology (toy, BBKS-based linear P(k))
h = 0.7
Omega_m = 0.3
n_s = 0.965
rho_crit = 2.775e11  # Msun/(Mpc/h)^3
rho_m = Omega_m * rho_crit

# k-grid and BBKS transfer
k = np.logspace(-3, 1.4, 1600)  # up to ~25 h/Mpc
Gamma = Omega_m * h
q = k / Gamma
T = np.log(1 + 2.34*q) / (2.34*q)
T *= (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4) ** (-0.25)
P_lcdm = (k**n_s) * (T**2)  # arbitrary amplitude

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

def R_of_M(M):
    return (3*M/(4*np.pi*rho_m))**(1/3)

def sigma_at_mass(Pk, M):
    R = R_of_M(M)
    x = k * R
    W = W_tophat(x)
    integrand = (k**2) * Pk * (W**2)
    sig2 = np.trapz(integrand, k) / (2*np.pi**2)
    return float(np.sqrt(max(sig2, 1e-30)))

def sigma_of_M_grid(Pk, M_grid):
    return np.array([sigma_at_mass(Pk, M) for M in M_grid])

delta_c = 1.686
def hmf_ps(M, sigma):
    lnM = np.log(M)
    ln_sig_inv = np.log(1.0/ sigma)
    dln_sig_inv_dlnM = np.gradient(ln_sig_inv, lnM)
    nu = delta_c / sigma
    f = np.sqrt(2/np.pi) * nu * np.exp(-0.5*nu**2)
    dn = (rho_m / M) * f * np.abs(dln_sig_inv_dlnM)
    dn = np.where(np.isfinite(dn) & (dn>0), dn, np.nan)
    return dn

def resonant_P(P_lcdm, tau, omega):
    return P_lcdm * np.exp(-tau * k**2) * (np.cos(omega * k)**2)

# Reference grids
M_grid = np.logspace(8.0, 12.0, 260)   # 1e8–1e12 Msun/h
sigma_LCDM_grid = sigma_of_M_grid(P_lcdm, M_grid)
dn_LCDM = hmf_ps(M_grid, sigma_LCDM_grid)

# Mass checkpoints for reporting
M9, M10, M11 = 1e9, 1e10, 1e11
sigma_LCDM_9  = sigma_at_mass(P_lcdm, M9)
sigma_LCDM_10 = sigma_at_mass(P_lcdm, M10)
sigma_LCDM_11 = sigma_at_mass(P_lcdm, M11)

# Mini-sweep ranges (as in the chat test)
taus   = np.linspace(0.05, 0.12, 24)
omegas = np.linspace(0.12, 0.20, 24)

rows = []
for tau in taus:
    for omega in omegas:
        P_res = resonant_P(P_lcdm, tau, omega)
        s9  = sigma_at_mass(P_res, M9)  / sigma_LCDM_9
        s10 = sigma_at_mass(P_res, M10) / sigma_LCDM_10
        s11 = sigma_at_mass(P_res, M11) / sigma_LCDM_11
        # score: prefer low s9/s10 and high s11
        penalty = 0.0
        if s11 < 0.90:
            penalty = 100.0 * (0.90 - s11)
        score = (s9 + 0.7*s10 - 0.5*s11) + penalty
        rows.append({"tau": float(tau), "omega": float(omega),
                     "sigma_ratio_1e9": float(s9), "sigma_ratio_1e10": float(s10),
                     "sigma_ratio_1e11": float(s11), "score": float(score)})

grid_df = pd.DataFrame(rows).sort_values("score")
grid_df.to_csv(f"{OUTDIR}/mini_sweep_tau_omega_ranked.csv", index=False)

top = grid_df[grid_df["sigma_ratio_1e11"]>=0.90].head(8)
top.to_csv(f"{OUTDIR}/mini_sweep_top8.csv", index=False)

if len(top) > 0:
    best = top.iloc[0].to_dict()
else:
    best = grid_df.iloc[0].to_dict()

tau_b  = float(best["tau"])
omega_b= float(best["omega"])

# Build best-case plots
P_res_b = resonant_P(P_lcdm, tau_b, omega_b)

# P(k)
plt.figure()
m1 = P_lcdm>0; m2 = P_res_b>0
plt.loglog(k[m1], P_lcdm[m1], label="P_LCDM")
plt.loglog(k[m2], P_res_b[m2], label=f"P_Res (tau={tau_b:.3f}, omega={omega_b:.3f})")
for n in range(0, 40):
    kn = (np.pi/2 + n*np.pi)/omega_b
    if k.min() < kn < k.max():
        plt.axvline(kn, linestyle="--", alpha=0.25)
plt.xlabel("k  [h/Mpc]")
plt.ylabel("P(k) [arb]")
plt.title("P(k): best mini-sweep candidate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/mini_pk_best.png", dpi=160)
plt.close()

# σ(M)
sigma_RES_grid = sigma_of_M_grid(P_res_b, M_grid)
plt.figure()
maskL = sigma_LCDM_grid>0; maskR = sigma_RES_grid>0
plt.loglog(M_grid[maskL], sigma_LCDM_grid[maskL], label="σ(M) LCDM")
plt.loglog(M_grid[maskR], sigma_RES_grid[maskR], label="σ(M) Resonant")
plt.xlabel("M  [Msun/h]")
plt.ylabel("σ(M)")
plt.title("Variance σ(M): best mini-sweep candidate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/mini_sigma_best.png", dpi=160)
plt.close()

# HMF
dn_RES_b = hmf_ps(M_grid, sigma_RES_grid)
plt.figure()
mL = np.isfinite(dn_LCDM) & (dn_LCDM>0)
mR = np.isfinite(dn_RES_b) & (dn_RES_b>0)
plt.loglog(M_grid[mL], dn_LCDM[mL], label="dn/dlnM LCDM")
plt.loglog(M_grid[mR], dn_RES_b[mR], label="dn/dlnM Resonant")
plt.xlabel("M  [Msun/h]")
plt.ylabel("dn/dlnM  [(Mpc/h)^{-3}]")
plt.title(f"HMF (tau={tau_b:.3f}, omega={omega_b:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/mini_hmf_best.png", dpi=160)
plt.close()

# Summary JSON
summary = {
    "best_params": {"tau": tau_b, "omega": omega_b},
    "best_sigma_ratios": {
        "1e9": float(best["sigma_ratio_1e9"]),
        "1e10": float(best["sigma_ratio_1e10"]),
        "1e11": float(best["sigma_ratio_1e11"])
    }
}
open(f"{OUTDIR}/mini_sweep_summary.json", "w").write(__import__("json").dumps(summary, indent=2))

print("Done. Outputs written to:", OUTDIR)
