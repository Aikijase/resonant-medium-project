import json, numpy as np
from halo_mass_function import linear_pk, sigma_grid
from utils_node_models import resonant_modifier, resonant_modifier_phase_smeared

DELTA_C = 1.686

def f_PS_of_sigma(sig):
    sig = np.clip(sig, 1e-12, 1e12)
    nu  = DELTA_C / sig
    return np.sqrt(2/np.pi) * nu * np.exp(-0.5 * nu**2) / sig  # ∝ f(σ); overall constants cancel in ratio

def estimate_ratio_PS_only(tau=0.05, omega=0.144,
                           Mmin=1e9, Mmax=1e10,
                           Omega_m=0.3, h=0.7, n_s=0.965,
                           phase_smooth=True):
    rho_crit = 2.775e11
    rho_m = Omega_m * rho_crit

    # k-grid and spectra
    k = np.logspace(-3, 1.4, 2000)
    P_lcdm = linear_pk(k, n_s=n_s, Omega_m=Omega_m, h=h)

    if phase_smooth:
        mod = resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=np.pi/20)
    else:
        mod = resonant_modifier(k, tau, omega, phase=0.0)
    P_res = P_lcdm * mod

    # Dense mass grid for stable integrals
    M = np.logspace(8, 12, 1600)
    lnM = np.log(M)
    win = (M >= Mmin) & (M <= Mmax)

    # σ(M)
    sigma_LCDM = sigma_grid(P_lcdm, k, M, rho_m)
    sigma_RES  = sigma_grid(P_res,  k, M, rho_m)

    # PS multiplicity weights (no derivative)
    wL = f_PS_of_sigma(sigma_LCDM)
    wR = f_PS_of_sigma(sigma_RES)

    # Integrate over the dwarf window
    m = win & np.isfinite(wL) & np.isfinite(wR)
    if m.sum() < 3:
        return {"suppression_ratio": None, "status": "not_enough_points"}

    IL = float(np.trapz(wL[m], lnM[m]))
    IR = float(np.trapz(wR[m], lnM[m]))

    ratio = None if IL<=0 else IR/IL
    return {
        "tau": tau, "omega": omega,
        "M_range_Msun_h": [float(Mmin), float(Mmax)],
        "phase_smooth": bool(phase_smooth),
        "PS_integral_LCDM": IL,
        "PS_integral_RES":  IR,
        "suppression_ratio": ratio
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=0.144)
    ap.add_argument("--Mmin", type=float, default=1e9)
    ap.add_argument("--Mmax", type=float, default=1e10)
    ap.add_argument("--no-phase-smooth", action="store_true")
    args = ap.parse_args()
    out = estimate_ratio_PS_only(tau=args.tau, omega=args.omega,
                                 Mmin=args.Mmin, Mmax=args.Mmax,
                                 phase_smooth=(not args.no_phase_smooth))
    print(json.dumps(out, indent=2))
