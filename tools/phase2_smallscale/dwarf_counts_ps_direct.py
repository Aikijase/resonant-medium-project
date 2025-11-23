import json, numpy as np
from halo_mass_function import linear_pk, sigma_grid
from utils_node_models import resonant_modifier, resonant_modifier_phase_smeared

DELTA_C = 1.686

def _smooth(y, w=11):
    y = np.asarray(y, float)
    w = int(max(3, w))
    if w % 2 == 0: w += 1
    p = w//2
    yy = np.pad(y, (p,p), mode='edge')
    k = np.ones(w)/w
    return np.convolve(yy, k, mode='valid')

def _stable_dlnsiginv_dlnM(M, sigma, win=31):
    M   = np.asarray(M, float)
    sig = np.clip(np.asarray(sigma, float), 1e-30, 1e30)
    lnM = np.log(M)
    y   = np.log(1.0/sig)
    ys  = _smooth(y, w=min(win, len(lnM)//2*2+1))
    d   = np.gradient(ys, lnM)
    m = np.isfinite(d)
    if not m.all():
        if m.any():
            d[~m] = np.interp(lnM[~m], lnM[m], d[m])
        else:
            d[:] = 0.0
    return d

def W_tophat(x):
    W = np.ones_like(x)
    small = x < 1e-4
    if np.any(small):
        xs = x[small]
        W[small] = 1 - xs**2/10.0
    big = ~small
    if np.any(big):
        xb = x[big]
        W[big] = 3*(np.sin(xb) - xb*np.cos(xb)) / (xb**3 + 1e-300)
    return W

def sigma_R_from_P(k, Pk, R):
    x = k*R
    W = W_tophat(x)
    sig2 = np.trapz((k**2)*Pk*(W**2), k) / (2*np.pi**2)
    return float(np.sqrt(max(sig2, 1e-30)))

def press_schechter_hmf(M, sigma, rho_m):
    # dn/dlnM = (rho_m/M) * f(ν) * |d ln σ^{-1} / d ln M|
    sig = np.clip(np.asarray(sigma, float), 1e-30, 1e30)
    nu  = DELTA_C / sig
    f   = np.sqrt(2/np.pi) * nu * np.exp(-0.5*nu**2)   # Press–Schechter multiplicity
    dln = _stable_dlnsiginv_dlnM(M, sig)
    dn  = (rho_m / M) * f * np.abs(dln)
    dn[~np.isfinite(dn)] = np.nan
    return dn

def estimate_counts_ps_direct(tau=0.05, omega=0.144,
                              Mmin=1e9, Mmax=1e10, Rvir_kpc_h=250.0,
                              Omega_m=0.3, h=0.7, n_s=0.965,
                              phase_smooth=True, sigma8_target=0.811):
    rho_crit = 2.775e11
    rho_m = Omega_m * rho_crit

    # k-grid and base P(k)
    k = np.logspace(-3, 1.4, 4000)  # slightly denser for σ8 accuracy
    P_lcdm = linear_pk(k, n_s=n_s, Omega_m=Omega_m, h=h)

    # --- σ8 normalization (R = 8 Mpc/h) ---
    R8 = 8.0  # (Mpc/h)
    sigma8_now = sigma_R_from_P(k, P_lcdm, R8)
    if sigma8_now <= 0:
        A = 1.0
    else:
        A = (sigma8_target / sigma8_now)**2
    P_lcdm *= A  # set σ8 to target

    # Resonant spectrum gets same amplitude rescale
    if phase_smooth:
        mod = resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=np.pi/20)
    else:
        mod = resonant_modifier(k, tau, omega, phase=0.0)
    P_res = P_lcdm * mod

    # Dense mass grid
    M = np.logspace(8, 12, 2000)
    lnM = np.log(M)

    sigma_LCDM = sigma_grid(P_lcdm, k, M, rho_m)
    sigma_RES  = sigma_grid(P_res,  k, M, rho_m)

    # HMF via PS (stable derivative inside this file)
    dnL = press_schechter_hmf(M, sigma_LCDM, rho_m)
    dnR = press_schechter_hmf(M, sigma_RES,  rho_m)

    # Integrate over the dwarf window
    win = (M >= Mmin) & (M <= Mmax)
    yL, yR = dnL[win], dnR[win]
    x       = lnM[win]

    mL = np.isfinite(yL); mR = np.isfinite(yR)
    nL = float(np.trapz(yL[mL], x[mL])) if mL.sum() >= 3 else float('nan')
    nR = float(np.trapz(yR[mR], x[mR])) if mR.sum() >= 3 else float('nan')

    # Host volume proxy
    V = (4/3)*np.pi*((Rvir_kpc_h/1000.0)**3)
    NL = None if not np.isfinite(nL) else nL * V
    NR = None if not np.isfinite(nR) else nR * V
    ratio = None if (NL is None or NL <= 0 or NR is None) else (NR/NL)

    return {
        "tau": tau, "omega": omega,
        "M_range_Msun_h": [Mmin, Mmax],
        "Rvir_kpc_h": Rvir_kpc_h,
        "sigma8_target": sigma8_target,
        "sigma8_before": sigma8_now,
        "amplitude_factor": A,
        "ndensity_LCDM_Mpch3": (None if not np.isfinite(nL) else nL),
        "ndensity_RES_Mpch3":  (None if not np.isfinite(nR) else nR),
        "N_expected_LCDM": NL,
        "N_expected_RES":  NR,
        "suppression_ratio": ratio,
        "finite_in_window": {
            "LCDM": int(np.isfinite(yL).sum()),
            "RES":  int(np.isfinite(yR).sum()),
            "total_bins": int(win.sum())
        }
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=0.144)
    ap.add_argument("--Mmin", type=float, default=1e9)
    ap.add_argument("--Mmax", type=float, default=1e10)
    ap.add_argument("--Rvir", type=float, default=250.0)
    ap.add_argument("--no-phase-smooth", action="store_true")
    ap.add_argument("--sigma8", type=float, default=0.811)
    args = ap.parse_args()
    out = estimate_counts_ps_direct(tau=args.tau, omega=args.omega,
                                    Mmin=args.Mmin, Mmax=args.Mmax, Rvir_kpc_h=args.Rvir,
                                    phase_smooth=(not args.no_phase_smooth),
                                    sigma8_target=args.sigma8)
    print(json.dumps(out, indent=2))
