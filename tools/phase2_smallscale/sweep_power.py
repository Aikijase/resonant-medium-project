# sweep_power.py
"""
Generate P(k), σ(M), and HMF comparisons for ΛCDM vs Resonant model,
with optional phase-smearing to soften wiggles.
"""
import os, json, numpy as np, matplotlib.pyplot as plt
from halo_mass_function import linear_pk, sigma_grid, hmf_press_schechter, hmf_sheth_tormen, hmf_tinker08
from halo_mass_function import bbks_transfer  # optional debug
from utils_node_models import resonant_modifier, resonant_modifier_phase_smeared

def run_sweep(outdir="outputs/phase2_smallscale",
              hmf_model="st",
              tau=0.05, omega=0.144, phase_smooth=True,
              Omega_m=0.3, h=0.7, n_s=0.965):
    os.makedirs(outdir, exist_ok=True)
    rho_crit = 2.775e11   # Msun / (Mpc/h)^3
    rho_m = Omega_m * rho_crit

    # k grid
    k = np.logspace(-3, 1.4, 1600)  # up to ~25 h/Mpc

    # Linear ΛCDM P(k)
    P_lcdm = linear_pk(k, n_s=n_s, Omega_m=Omega_m, h=h)

    # Resonant modifier
    if phase_smooth:
        mod = resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=np.pi/20)
    else:
        mod = resonant_modifier(k, tau, omega, phase=0.0)

    P_res = P_lcdm * mod

    # M grid and σ(M)
    M_grid = np.logspace(8.0, 12.0, 260)  # 1e8–1e12 Msun/h

    sigma_LCDM = sigma_grid(P_lcdm, k, M_grid, rho_m)
    sigma_RES  = sigma_grid(P_res,  k, M_grid, rho_m)

    # HMF selection
    if hmf_model.lower() in ["ps", "press", "press-schechter"]:
        hmf_func = hmf_press_schechter
        hmf_name = "Press–Schechter"
    elif hmf_model.lower() in ["st", "sheth", "sheth-tormen"]:
        hmf_func = hmf_sheth_tormen
        hmf_name = "Sheth–Tormen"
    elif hmf_model.lower() in ["tinker", "tinker08", "t08"]:
        hmf_func = hmf_tinker08
        hmf_name = "Tinker+08"
    else:
        raise ValueError("Unknown hmf_model; use one of: ps, st, tinker")

    dn_LCDM = hmf_func(M_grid, sigma_LCDM, rho_m)
    dn_RES  = hmf_func(M_grid, sigma_RES,  rho_m)

    # Key masses
    targets = [1e9, 1e10, 1e11]
    def sigma_ratios_at_targets():
        out = {}
        for Mt in targets:
            i = int(np.argmin(np.abs(M_grid - Mt)))
            if sigma_LCDM[i] > 0:
                out[str(Mt)] = float(sigma_RES[i]/sigma_LCDM[i])
            else:
                out[str(Mt)] = None
        return out

    ratios = sigma_ratios_at_targets()

    # Save JSON summary
    summary = dict(
        model=dict(tau=tau, omega=omega, phase_smooth=bool(phase_smooth)),
        hmf=hmf_name,
        sigma_ratio_at_M=ratios
    )
    open(os.path.join(outdir, "summary.json"), "w").write(json.dumps(summary, indent=2))

    # Plots
    # 1) P(k)
    plt.figure()
    mask = P_lcdm>0
    plt.loglog(k[mask], P_lcdm[mask], label="P_LCDM")
    mask2 = P_res>0
    plt.loglog(k[mask2], P_res[mask2], label=f"P_Res (τ={tau:.3f}, ω={omega:.3f})")
    # draw node guides for unsmeared case (approx)
    for n in range(0, 40):
        kn = (np.pi/2 + n*np.pi)/omega
        if k.min() < kn < k.max():
            plt.axvline(kn, linestyle="--", alpha=0.25)
    plt.xlabel("k  [h/Mpc]")
    plt.ylabel("P(k) [arb]")
    plt.title("Linear Power Spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pk_compare.png"), dpi=160)
    plt.close()

    # 2) σ(M)
    plt.figure()
    mL = sigma_LCDM>0; mR = sigma_RES>0
    plt.loglog(M_grid[mL], sigma_LCDM[mL], label="σ(M) LCDM")
    plt.loglog(M_grid[mR], sigma_RES[mR],   label="σ(M) Resonant")
    plt.xlabel("M  [Msun/h]")
    plt.ylabel("σ(M)")
    plt.title("Variance σ(M)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sigma_compare.png"), dpi=160)
    plt.close()

    # 3) HMF
    plt.figure()
    m1 = np.isfinite(dn_LCDM) & (dn_LCDM>0)
    m2 = np.isfinite(dn_RES)  & (dn_RES>0)
    plt.loglog(M_grid[m1], dn_LCDM[m1], label=f"dn/dlnM LCDM ({hmf_name})")
    plt.loglog(M_grid[m2], dn_RES[m2],  label=f"dn/dlnM Resonant ({hmf_name})")
    plt.xlabel("M  [Msun/h]")
    plt.ylabel("dn/dlnM  [(Mpc/h)^{-3}]")
    plt.title(f"Halo Mass Function — {hmf_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hmf_compare.png"), dpi=160)
    plt.close()

    return summary

if __name__ == "__main__":
    # simple CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/phase2_smallscale")
    ap.add_argument("--hmf", default="st", help="ps | st | tinker")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=0.144)
    ap.add_argument("--no-phase-smooth", action="store_true", help="Disable phase-smearing of nodes")
    args = ap.parse_args()
    summary = run_sweep(outdir=args.outdir,
                        hmf_model=args.hmf,
                        tau=args.tau, omega=args.omega,
                        phase_smooth=(not args.no_phase_smooth))
    print(json.dumps(summary, indent=2))
