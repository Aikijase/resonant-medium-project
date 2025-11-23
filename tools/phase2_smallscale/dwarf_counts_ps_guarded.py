import json, numpy as np
from halo_mass_function import linear_pk, sigma_grid
from mod_guard import resonant_modifier_guarded

DELTA_C = 1.686

def _smooth(y, w=11):
    y = np.asarray(y, float)
    w = int(max(3, w));  w += (w % 2 == 0)
    p = w//2
    yy = np.pad(y, (p,p), mode='edge')
    return np.convolve(yy, np.ones(w)/w, mode='valid')

def _stable_dlnsiginv_dlnM(M, sigma, win=31):
    M = np.asarray(M); lnM = np.log(M)
    sig = np.clip(np.asarray(sigma, float), 1e-30, 1e30)
    ys = _smooth(np.log(1.0/sig), w=min(win, len(lnM)//2*2+1))
    d  = np.gradient(ys, lnM)
    m = np.isfinite(d)
    if not m.all():
        d[~m] = np.interp(lnM[~m], lnM[m], d[m]) if m.any() else 0.0
    return d

def press_schechter_hmf(M, sigma, rho_m):
    sig = np.clip(np.asarray(sigma, float), 1e-30, 1e30)
    nu  = DELTA_C / sig
    f   = np.sqrt(2/np.pi) * nu * np.exp(-0.5*nu**2)
    dln = _stable_dlnsiginv_dlnM(M, sig)
    dn  = (rho_m / M) * f * np.abs(dln)
    dn[~np.isfinite(dn)] = np.nan
    return dn

def W_tophat(x):
    W = np.ones_like(x)
    small = x < 1e-4
    if np.any(small): W[small] = 1 - x[small]**2/10.0
    xb = x[~small]
    if xb.size: W[~small] = 3*(np.sin(xb) - xb*np.cos(xb)) / (xb**3 + 1e-300)
    return W

def sigma_R_from_P(k, Pk, R):
    x = k*R; W = W_tophat(x)
    sig2 = np.trapz((k**2)*Pk*(W**2), k)/(2*np.pi**2)
    return float(np.sqrt(max(sig2, 1e-30)))

def estimate_counts_ps_guarded(tau=0.05, omega=0.144,
                               Mmin=1e9, Mmax=1e10, Rvir_kpc_h=250.0,
                               Omega_m=0.3, h=0.7, n_s=0.965,
                               sigma8_target=0.811,
                               k_guard=8.0, delta=0.7, n_phase=9, width=np.pi/8):
    rho_crit = 2.775e11; rho_m = Omega_m * rho_crit

    # k-grid and base P(k)
    k = np.logspace(-3, 1.4, 4000)
    P_lcdm = linear_pk(k, n_s=n_s, Omega_m=Omega_m, h=h)

    # σ8 normalization
    R8 = 8.0
    s8_now = sigma_R_from_P(k, P_lcdm, R8)
    A = 1.0 if s8_now <= 0 else (sigma8_target/s8_now)**2
    P_lcdm *= A

    # Guarded resonant modifier (Ly-α safe)
    mod = resonant_modifier_guarded(k, tau, omega, n_phase=n_phase, width=width,
                                    k_guard=k_guard, delta=delta)
    P_res = P_lcdm * mod

    # Mass grid & HMF (PS)
    M = np.logspace(8, 12, 2000); lnM = np.log(M)
    sigL = sigma_grid(P_lcdm, k, M, rho_m)
    sigR = sigma_grid(P_res,  k, M, rho_m)
    dnL  = press_schechter_hmf(M, sigL, rho_m)
    dnR  = press_schechter_hmf(M, sigR, rho_m)

    # Integrate over dwarf window
    win = (M >= Mmin) & (M <= Mmax)
    def integ(dn):
        y = dn[win]; x = lnM[win]; m = np.isfinite(y)
        return float(np.trapz(y[m], x[m])) if m.sum() >= 3 else float('nan')
    nL = integ(dnL); nR = integ(dnR)

    # Host volume proxy
    V = (4/3)*np.pi*((Rvir_kpc_h/1000.0)**3)
    NL = None if not np.isfinite(nL) else nL*V
    NR = None if not np.isfinite(nR) else nR*V
    ratio = None if (NL is None or NL <= 0 or NR is None) else (NR/NL)

    return {
        "tau": tau, "omega": omega,
        "M_range_Msun_h": [Mmin, Mmax],
        "Rvir_kpc_h": Rvir_kpc_h,
        "sigma8_before": s8_now,
        "amplitude_factor": A,
        "suppression_ratio": ratio,
        "lowk_guard": {"k_guard": k_guard, "delta": delta, "n_phase": n_phase, "width": float(width)}
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=0.144)
    ap.add_argument("--Mmin", type=float, default=1e9)
    ap.add_argument("--Mmax", type=float, default=1e10)
    ap.add_argument("--Rvir", type=float, default=250.0)
    ap.add_argument("--sigma8", type=float, default=0.811)
    ap.add_argument("--k_guard", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=0.7)
    args = ap.parse_args()
    out = estimate_counts_ps_guarded(tau=args.tau, omega=args.omega,
                                     Mmin=args.Mmin, Mmax=args.Mmax, Rvir_kpc_h=args.Rvir,
                                     sigma8_target=args.sigma8,
                                     k_guard=args.k_guard, delta=args.delta)
    print(json.dumps(out, indent=2))
