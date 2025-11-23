#!/usr/bin/env python3
# tools/isw_forecast.py
# ISW forecast: Planck TT + tracer n(z), b(z), Limber (+ optional non-Limber at low ell), shot noise, tomography.

import os, csv, argparse, numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from scipy import integrate
from scipy.special import spherical_jn

# ---------- Constants (flat LCDM defaults; override on CLI if you like) ----------
Tcmb_uK = 2.7255e6  # μK
c_kms    = 299792.458
c_Mpc_s  = 9.71561189e-15  # ~ c / (1 Mpc) in s^-1, not used directly
H0       = 67.4            # km/s/Mpc
h        = H0/100.0
Omega_m  = 0.315
Omega_L  = 1.0 - Omega_m
ns       = 0.965
As_2e9   = 2.1  # A_s × 1e9, crude fallback if you don't load external P(k)

# ---------- Numerics ----------
ELL_MIN, ELL_MAX = 2, 100
NZ_GRID          = 800
K_MIN, K_MAX     = 1e-4, 0.5  # Mpc^-1 for non-Limber integral
K_N              = 450

# ---------- Helpers ----------
def Ez(z):  # H(z)/H0
    return np.sqrt(Omega_m*(1+z)**3 + Omega_L)

def chi_of_z(z):  # comoving distance [Mpc]
    z = np.atleast_1d(z)
    f = lambda zp: c_kms / (H0*Ez(zp))
    return np.array([integrate.quad(f, 0.0, zi, epsabs=0, epsrel=3e-6)[0] for zi in z])

def growth_D(a):
    """ Carroll, Press & Turner 1992 approx. Normalized to D(a=1)=1. """
    Om_a = Omega_m/(Omega_m + Omega_L*a**3)
    Ol_a = 1.0 - Om_a
    g = 5.0*Om_a/(2.0*(Om_a**(4/7) - Ol_a + (1+0.5*Om_a)*(1 + Ol_a/70.0)))
    return g/(g[ -1 ])  # normalize at a=1

def linear_Pk_EH_nw(k, z):
    """ Very crude Eisenstein-Hu 'no-wiggle' style P(k,z) ∝ k^ns T^2(k) D^2(z).
        This is good enough for ISW shapes; you’ll typically supply a CAMB/CLASS P(k) if you want better fidelity. """
    # Simple transfer proxy: T(k) ~ (1 + (a k + (b k)**1.5 + (c k)**2)**nu )**(-1/nu)  (smooth roll-off)
    # Choose numbers to mimic the turnover scale ~ k ~ 0.02-0.03 h/Mpc.
    kh     = k/h
    a, b, c, nu = 6.0, 3.0, 1.7, 1.2
    Tk     = (1.0 + (a*kh + (b*kh)**1.5 + (c*kh)**2.0)**nu )**(-1.0/nu)
    a_z    = 1.0/(1.0+z)
    Dz     = growth_D(np.array([a_z]))[0]
    P0     = 1e4  # arbitrary normalization (forecast uses amplitude ratios)
    return P0 * (k**ns) * (Tk**2) * (Dz**2)

def load_nz_bias(path: str, zmax_hint=2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(path):
        z, nz, b = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
        # normalize nz to ∫dz nz = 1
        norm = np.trapz(nz, z)
        if norm > 0: nz = nz / norm
        return z, nz, b
    # Fabricate a decent wide survey as a fallback (replace with your file!)
    z = np.linspace(0.01, zmax_hint, 400)
    z0 = 0.35; alpha=2.0; beta=2.0
    nz = (z/z0)**alpha * np.exp(-(z/z0)**beta); nz /= np.trapz(nz, z)
    b  = 1.0 + 0.8*z
    return z, nz, b

def make_bins(z, nz, edges: List[float]):
    """ Top-hat bins in z with nz renormalized inside each bin; returns list of (z_i, nz_i). """
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z>=lo) & (z<hi)
        if not np.any(m):  # empty bin → skip
            out.append((None,None))
            continue
        zi, nzi = z[m], nz[m].copy()
        norm = np.trapz(nzi, zi)
        if norm>0: nzi /= norm
        out.append((zi, nzi))
    return out

def dlnD_deta(z):
    """ ISW kernel factor ~ d/dη(D/a); here we approximate w.r.t conformal time via chain rule.
        A practical proxy (literature shows the sign/shape is what matters for forecast S/N):
        d(D/a)/dη = a H(a) d(D/a)/d(a)  with d/da = d/d(ln a) * 1/a.
    """
    a = 1.0/(1.0+z)
    D = growth_D(a)
    # finite diff in ln a
    eps = 1e-3
    ap  = a*(1+eps); am = a*(1-eps)
    Dp  = growth_D(ap); Dm = growth_D(am)
    ln_deriv = ( (Dp/ap) - (Dm/am) )/(np.log(ap)-np.log(am))
    # convert to d/dη = aH * d/d(ln a)
    Hz = H0*Ez(z)  # km/s/Mpc
    # We only need shape → rescale to (km/s/Mpc) cancels in template amplitude.
    return a*Hz*ln_deriv

def load_ClTT_planck(path: str, ells):
    if os.path.exists(path):
        ell_tab, CTT_tab = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
        # ensure ℓ range covers desired band
        return np.interp(ells, ell_tab, CTT_tab)
    # Fallback envelope (decays ~ ℓ^-2 for low ℓ after Sachs-Wolfe plateau)
    ell0=10.0
    A   = 1.1e3  # μK^2 at ℓ~10
    return A*(ell0/np.maximum(ells,2))**2

@dataclass
class Inputs:
    z: np.ndarray
    nz: np.ndarray
    b: np.ndarray
    chi: np.ndarray
    Wg: np.ndarray
    WT: np.ndarray

def build_windows(z, nz, b) -> Inputs:
    chi = chi_of_z(z)
    a   = 1.0/(1.0+z)
    D   = growth_D(a)
    # Galaxy window (Limber): Wg(z) = b(z) * nz(z) * D(z)
    Wg  = b * nz * D
    # ISW temperature window: WT(z) ∝ d/dη(D/a)  (shape only; absolute normalization cancels in A-fit)
    WT  = dlnD_deta(z)
    return Inputs(z=z, nz=nz, b=b, chi=chi, Wg=Wg, WT=WT)

def Cl_Tg_limber(ell, inp: Inputs):
    """ Limber approximation: C_l^{Tg} ≈ ∫ dz (1/chi^2) WT(z) Wg(z) P(k=(ℓ+1/2)/χ, z) / H(z) """
    z, chi, Wg, WT = inp.z, inp.chi, inp.Wg, inp.WT
    ell_eff = ell + 0.5
    k = ell_eff/np.maximum(chi, 1e-6)
    Pk = linear_Pk_EH_nw(k, z)
    Hz = H0*Ez(z)
    integrand = WT*Wg*Pk/(chi**2 * Hz)
    return integrate.simpson(integrand, x=z)

def Cl_Tg_nonlimber(ell, inp: Inputs):
    """ Quick-and-clean non-Limber using spherical Bessel projection at low ℓ. """
    z, chi, Wg, WT = inp.z, inp.chi, inp.Wg, inp.WT
    k  = np.logspace(np.log10(K_MIN), np.log10(K_MAX), K_N)
    # project windows to I_ell(k)
    j = spherical_jn(ell, np.outer(k, chi))  # [Nk, Nz]
    IT = integrate.simpson((WT * j)/(np.maximum(chi,1e-12)**2), x=z, axis=1)  # [Nk]
    Ig = integrate.simpson((Wg * j)/(np.maximum(chi,1e-12)**2), x=z, axis=1)  # [Nk]
    # Use z~0.3 as representative for P(k); you can refine with z(k) mapping if you like
    Pk = linear_Pk_EH_nw(k, 0.3)
    return integrate.simpson(Pk * IT * Ig, x=np.log(k))  # log-integral for stability

def forecast_SN(ells, ClTT, Clgg, ClTg, f_sky):
    var = (ClTT*Clgg + ClTg**2) / ((2*ells+1)*f_sky)
    SNl = ClTg/np.sqrt(np.maximum(var, 1e-300))
    SNcum = np.sqrt(np.cumsum(SNl**2))
    return SNl, SNcum

def galaxy_auto_limber(ell, inp: Inputs, nbar_sr=None):
    """ C_l^{gg} (signal+shot). Limber with same k=(ℓ+1/2)/χ. Shot noise = 1/n̄ (sr). """
    z, chi, Wg = inp.z, inp.chi, inp.Wg
    ell_eff = ell + 0.5
    k  = ell_eff/np.maximum(chi, 1e-6)
    Pk = linear_Pk_EH_nw(k, z)
    Hz = H0*Ez(z)
    # note: Wg contains nz already; for autos you usually need (nz*b*D)^2/H/chi^2
    integrand = (Wg**2)*Pk/(chi**2 * Hz)
    Clgg = integrate.simpson(integrand, x=z)
    if nbar_sr is not None and nbar_sr>0:
        Clgg += 1.0/nbar_sr
    return Clgg

def arcmin2_to_sr(nbar_arcmin2):
    return nbar_arcmin2 * (60.0*np.pi/10800.0)**2  # (π/180/60)^2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz-bias", default="data/nz_bias.csv", help="CSV: z,nz,b")
    ap.add_argument("--cltt", default="data/planck_2018_TT_ell_2_200.csv", help="CSV: ell,CTT_uK2")
    ap.add_argument("--fsky", type=float, default=0.30)
    ap.add_argument("--nbar-arcmin2", type=float, default=0.50)
    ap.add_argument("--ell-max", type=int, default=ELL_MAX)
    ap.add_argument("--nonlimber", action="store_true", help="Use non-Limber for ℓ≤20")
    ap.add_argument("--bins", default="", help="Comma sep. edges like '0.0,0.3,0.6,1.0'. Blank = no tomography.")
    ap.add_argument("--out-prefix", default="outputs/isw", help="Output prefix")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # Load inputs
    z_all, nz_all, b_all = load_nz_bias(args.nz_bias)
    ells = np.arange(ELL_MIN, args.ell_max+1)
    ClTT_uK2 = load_ClTT_planck(args.cltt, ells)

    # Tomography bins
    if args.bins.strip():
        edges = [float(x) for x in args.bins.split(",")]
        bins = make_bins(z_all, nz_all, edges)
    else:
        bins = [(z_all, nz_all)]

    # Shot noise
    nbar_sr = arcmin2_to_sr(args.nbar_arcmin2)

    # Compute per-bin spectra and S/N
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.5,5.0))

    colors = []
    ClTg_bins = []
    SNcum_bins = []

    # For the “no-bin” case, reuse b(z); for binned case, interpolate b(z) onto each bin’s grid.
    for i, (zi, nzi) in enumerate(bins):
        if zi is None:  # empty
            continue
        bi = np.interp(zi, z_all, b_all)
        inp = build_windows(zi, nzi, bi)

        ClTg = np.zeros_like(ells, dtype=float)
        Clgg = np.zeros_like(ells, dtype=float)
        for j, L in enumerate(ells):
            if args.nonlimber and L<=20:
                ClTg[j] = Cl_Tg_nonlimber(L, inp)
            else:
                ClTg[j] = Cl_Tg_limber(L, inp)
            Clgg[j] = galaxy_auto_limber(L, inp, nbar_sr=nbar_sr)

        SNl, SNcum = forecast_SN(ells, ClTT_uK2, Clgg, ClTg, args.fsky)
        ClTg_bins.append(ClTg)
        SNcum_bins.append(SNcum)

        # Plot C_l^{Tg}
        plt.plot(ells, ClTg, label=f"bin {i+1}")

    # Sum S/N in quadrature across bins
    if len(SNcum_bins)==0:
        total_SN = 0.0
        SNcum_tot = np.zeros_like(ells, dtype=float)
    elif len(SNcum_bins)==1:
        SNcum_tot = SNcum_bins[0]
        total_SN  = SNcum_tot[-1]
    else:
        # Recompute per-ℓ S/N, then quadrature sum across bins
        # (We stored only cum; recompute properly:)
        # Do it right: recompute SNl per bin from stored ClTg, then sum.
        # We need Clgg per bin again; recompute swiftly:
        SNl_bins = []
        for (zi,nzi), ClTg in zip(bins, ClTg_bins):
            bi = np.interp(zi, z_all, b_all)
            inp = build_windows(zi, nzi, bi)
            Clgg = np.array([galaxy_auto_limber(L, inp, nbar_sr) for L in ells])
            var  = (ClTT_uK2*Clgg + ClTg**2)/((2*ells+1)*args.fsky)
            SNl_bins.append(ClTg/np.sqrt(np.maximum(var,1e-300)))
        SNl_tot  = np.sqrt(np.sum(np.array(SNl_bins)**2, axis=0))
        SNcum_tot= np.sqrt(np.cumsum(SNl_tot**2))
        total_SN = SNcum_tot[-1]

    # Final plot embellishments
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell^{Tg}$  (arb. units)")
    ttl = f"ISW forecast (f_sky={args.fsky:.2f}, nbar={args.nbar_arcmin2:.3f} arcmin$^{{-2}}$"
    ttl+= ", non-Limber≤20" if args.nonlimber else ", Limber"
    ttl+= f") — total S/N≈{total_SN:.2f}"
    plt.title(ttl)
    plt.legend(frameon=False)
    fig_path = f"{args.out_prefix}_tg_with_sn.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)

    # Write per-ℓ cumulative S/N
    with open(f"{args.out_prefix}_sn_table.csv","w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell","SN_cum"])
        for L, s in zip(ells, SNcum_tot):
            w.writerow([int(L), float(s)])

    # Quick stdout summary
    print(f"[isw_forecast] wrote {fig_path}")
    print(f"[isw_forecast] total S/N ≈ {total_SN:.2f}")
    print(f"[isw_forecast] per-ℓ cum S/N → {args.out_prefix}_sn_table.csv")

if __name__ == "__main__":
    main()
