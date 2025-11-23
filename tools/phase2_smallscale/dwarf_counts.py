import json, numpy as np
from halo_mass_function import linear_pk, sigma_grid, hmf_press_schechter, hmf_sheth_tormen, hmf_tinker08
from utils_node_models import resonant_modifier, resonant_modifier_phase_smeared

def _finite_integral(dn, lnM, mask):
    y = dn[mask]; x = lnM[mask]
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 4:  # need at least a few points
        return None
    return float(np.trapz(y[m], x[m]))

def _hmf_dispatch(name):
    name = name.lower()
    if name in ("tinker","tinker08","t08"):
        return hmf_tinker08, "Tinker+08"
    if name in ("st","sheth","sheth-tormen"):
        return hmf_sheth_tormen, "Sheth–Tormen"
    return hmf_press_schechter, "Press–Schechter"

def estimate_counts(hmf_model="tinker", tau=0.05, omega=0.144,
                    Mmin=1e9, Mmax=1e10, Rvir_kpc_h=250.0,
                    Omega_m=0.3, h=0.7, n_s=0.965, phase_smooth=True):
    rho_crit = 2.775e11
    rho_m = Omega_m * rho_crit

    # k-grid + spectra
    k = np.logspace(-3, 1.4, 2000)
    P_lcdm = linear_pk(k, n_s=n_s, Omega_m=Omega_m, h=h)
    mod = (resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=np.pi/20)
           if phase_smooth else
           resonant_modifier(k, tau, omega, phase=0.0))
    P_res = P_lcdm * mod

    # Denser M-grid for a stable derivative
    M = np.logspace(8, 12, 1600)
    sigma_LCDM = sigma_grid(P_lcdm, k, M, rho_m)
    sigma_RES  = sigma_grid(P_res,  k, M, rho_m)
    lnM = np.log(M); window = (M >= Mmin) & (M <= Mmax)

    # Try preferred → fallbacks
    order = [hmf_model, "st", "ps"]
    picked_name = None
    n_LCDM = n_RES = None

    for choice in order:
        hmf_fn, hmf_name = _hmf_dispatch(choice)
        dnL = hmf_fn(M, sigma_LCDM, rho_m)
        dnR = hmf_fn(M, sigma_RES,  rho_m)
        nL = _finite_integral(dnL, lnM, window)
        nR = _finite_integral(dnR, lnM, window)
        if (nL is not None) and (nR is not None):
            picked_name, n_LCDM, n_RES = hmf_name, nL, nR
            break

    # Host volume
    R_Mpch = Rvir_kpc_h / 1000.0
    V = (4/3)*np.pi*(R_Mpch**3)

    N_LCDM = None if n_LCDM is None else n_LCDM * V
    N_RES  = None if n_RES  is None else n_RES  * V
    supp = (None if (N_LCDM is None or N_LCDM <= 0 or N_RES is None)
            else (N_RES / N_LCDM))

    return {
        "hmf": picked_name or "none",
        "tau": tau, "omega": omega,
        "M_range_Msun_h": [Mmin, Mmax],
        "Rvir_kpc_h": Rvir_kpc_h,
        "ndensity_LCDM_Mpch3": (float(n_LCDM) if n_LCDM is not None else None),
        "ndensity_RES_Mpch3":  (float(n_RES)  if n_RES  is not None else None),
        "N_expected_LCDM": (float(N_LCDM) if N_LCDM is not None else None),
        "N_expected_RES":  (float(N_RES)  if N_RES  is not None else None),
        "suppression_ratio": (float(supp) if supp is not None else None)
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmf", default="tinker", help="ps|st|tinker (will fallback if needed)")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=0.144)
    ap.add_argument("--Mmin", type=float, default=1e9)
    ap.add_argument("--Mmax", type=float, default=1e10)
    ap.add_argument("--Rvir", type=float, default=250.0)
    ap.add_argument("--no-phase-smooth", action="store_true")
    args = ap.parse_args()
    out = estimate_counts(hmf_model=args.hmf, tau=args.tau, omega=args.omega,
                          Mmin=args.Mmin, Mmax=args.Mmax, Rvir_kpc_h=args.Rvir,
                          phase_smooth=(not args.no_phase_smooth))
    print(json.dumps(out, indent=2))
