#!/usr/bin/env python3
# tools/isw_xcorr_maps.py
#
# Planck×LSS ISW cross-power, errors, cumulative S/N, and amplitude fit A_ISW.
# - Inputs: T map (μK), δg map(s), mask (boolean or 0/1), all same NSIDE / coords.
# - Cross-spectra via healpy (pseudo-Cl) or NaMaster (MASTER debias) if --namaster.
# - Theory template: either recomputed from nz,b (imports a minimal copy here) or loaded from CSV.
#
# Outputs:
#   outputs/isw_maps_cl_tg_[tag].csv        (ℓ, Cl_Tg, σ_ℓ, SN_ℓ, SN_cum)
#   outputs/isw_maps_summary_[tag].txt      (S/N_tot, A±σ_A, fit χ²/dof)
#   outputs/isw_maps_tg_plot_[tag].png      (data + theory×A_fit; per-bin if provided)
#
import os, sys, csv, argparse, numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# -------------------- Optional NaMaster --------------------
try:
    import pymaster as nmt
    HAS_NM = True
except Exception:
    HAS_NM = False

# -------------------- Small cosmology toolbox (matches forecast A) --------------------
from scipy import integrate
from scipy.special import spherical_jn

H0 = 67.4
Omega_m = 0.315
Omega_L = 1.0 - Omega_m

def Ez(z): return np.sqrt(Omega_m*(1+z)**3 + Omega_L)

def chi_of_z(z):
    f = lambda zp: 299792.458 / (H0*Ez(zp))
    return np.array([integrate.quad(f, 0.0, zi, epsrel=3e-6)[0] for zi in np.atleast_1d(z)])

def growth_D(a):
    Om_a = Omega_m/(Omega_m + Omega_L*a**3)
    Ol_a = 1.0 - Om_a
    g = 5.0*Om_a/(2.0*(Om_a**(4/7) - Ol_a + (1+0.5*Om_a)*(1 + Ol_a/70.0)))
    return g/g[-1]

def dlnD_deta(z):
    a = 1.0/(1.0+z)
    eps = 1e-3
    ap, am = a*(1+eps), a*(1-eps)
    Dp, Dm = growth_D(ap), growth_D(am)
    ln_deriv = ( (Dp/ap) - (Dm/am) )/(np.log(ap)-np.log(am))
    Hz = H0*Ez(z)
    return a*Hz*ln_deriv

def linear_Pk_EH_nw(k, z):
    # Smooth, no-wiggle proxy (amplitude cancels in A-fit)
    h = H0/100.0
    kh = k/h
    a, b, c, nu = 6.0, 3.0, 1.7, 1.2
    Tk = (1.0 + (a*kh + (b*kh)**1.5 + (c*kh)**2.0)**nu )**(-1.0/nu)
    ns = 0.965
    a_z = 1.0/(1.0+z); Dz = growth_D(np.array([a_z]))[0]
    return 1e4 * (k**ns) * (Tk**2) * (Dz**2)

def build_windows(z, nz, b):
    chi = chi_of_z(z)
    a   = 1.0/(1.0+z)
    D   = growth_D(a)
    Wg  = b * nz * D
    WT  = dlnD_deta(z)
    return dict(z=z, nz=nz, b=b, chi=chi, Wg=Wg, WT=WT)

def Cl_Tg_limber(ell, inp):
    z, chi, Wg, WT = inp['z'], inp['chi'], inp['Wg'], inp['WT']
    k  = (ell+0.5)/np.maximum(chi,1e-6)
    Pk = linear_Pk_EH_nw(k, z)
    Hz = H0*Ez(z)
    integrand = WT*Wg*Pk/(chi**2 * Hz)
    return integrate.simpson(integrand, x=z)

def galaxy_auto_limber(ell, inp, nbar_sr=None):
    z, chi, Wg = inp['z'], inp['chi'], inp['Wg']
    k  = (ell+0.5)/np.maximum(chi,1e-6)
    Pk = linear_Pk_EH_nw(k, z)
    Hz = H0*Ez(z)
    Cl = integrate.simps((Wg**2)*Pk/(chi**2*Hz), z)
    if nbar_sr and nbar_sr>0: Cl += 1.0/nbar_sr
    return Cl

def load_nz_bias(path, zmax_hint=2.0):
    if path and os.path.exists(path):
        z, nz, b = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
        nz = nz/np.trapz(nz, z)
        return z, nz, b
    # fallback
    z = np.linspace(0.01, zmax_hint, 400)
    z0=0.35; alpha=2.0; beta=2.0
    nz=(z/z0)**alpha*np.exp(-(z/z0)**beta); nz/=np.trapz(nz,z)
    b=1+0.8*z
    return z, nz, b

def arcmin2_to_sr(nbar_arcmin2):
    return nbar_arcmin2 * (60.0*np.pi/10800.0)**2

# -------------------- IO helpers --------------------
def read_map(path):
    m = hp.read_map(path)
    return np.asarray(m, dtype=float)

def ensure_same_nside(maps):
    ns = [hp.get_nside(m) for m in maps]
    if len(set(ns))!=1:
        raise ValueError(f"All maps must share NSIDE; got {ns}")
    return ns[0]

def load_theory_template(ells, template_csv=None, nz_bias=None, nbar_arcmin2=None):
    """
    Returns theory Cl_Tg(ℓ).
    Priority: if template_csv provided → load column 'Cl_Tg'. Else recompute from nz,b.
    """
    if template_csv and os.path.exists(template_csv):
        L, Ct = np.loadtxt(template_csv, delimiter=",", skiprows=1, unpack=True)
        return np.interp(ells, L, Ct)

    if nz_bias:
        z, nz, b = load_nz_bias(nz_bias)
        inp = build_windows(z, nz, b)
        # shot for gg only; theory template Cl_Tg unaffected by nbar
        return np.array([Cl_Tg_limber(L, inp) for L in ells])

    raise ValueError("Need either --template-csv or --nz-bias to build theory.")

# -------------------- Spectra & variance --------------------
def compute_cls_anafast(Tmap, gmap, mask, lmax):
    m = mask.astype(float)
    Cl_Tg = hp.anafast(Tmap*m, gmap*m, lmax=lmax)
    Cl_TT = hp.anafast(Tmap*m, lmax=lmax)
    Cl_gg = hp.anafast(gmap*m, lmax=lmax)
    fsky  = np.mean(mask)
    return Cl_Tg, Cl_TT, Cl_gg, fsky

def compute_cls_namaster(Tmap, gmap, mask, lmax, nlb=8):
    if not HAS_NM:
        raise RuntimeError("pymaster not available; omit --namaster or install NaMaster.")
    # Binning scheme
    b = nmt.NmtBin.from_nside_linear(hp.get_nside(mask), nlb, is_Dell=False, lmax=lmax)
    # Fields with mask; no spins
    fT = nmt.NmtField(mask, [Tmap])
    fg = nmt.NmtField(mask, [gmap])
    # Coupled→decoupled spectra
    cl_coupled = nmt.compute_coupled_cell(fT, fg)[0]
    w = nmt.NmtWorkspace(); w.compute_coupling_matrix(fT, fg, b)
    cl_dec = w.decouple_cell(cl_coupled)
    # Auto-spectra for errors
    clTT_c = nmt.compute_coupled_cell(fT, fT)[0]
    clgg_c = nmt.compute_coupled_cell(fg, fg)[0]
    clTT = w.decouple_cell(clTT_c); clgg = w.decouple_cell(clgg_c)
    ells = b.get_effective_ells()
    fsky = np.mean(mask)
    return ells, cl_dec, clTT, clgg, fsky

def variance_and_sn(ells, ClTT, Clgg, ClTg, fsky):
    var = (ClTT*Clgg + ClTg**2) / ((2*ells+1)*fsky)
    SNl = ClTg/np.sqrt(np.maximum(var, 1e-300))
    SNcum = np.sqrt(np.cumsum(SNl**2))
    sig = np.sqrt(np.maximum(var, 1e-300))
    return sig, SNl, SNcum

def fit_amplitude(Cl_data, Cl_theory, sigma):
    W = 1.0/np.maximum(sigma, 1e-300)**2
    num = np.sum(W * Cl_data * Cl_theory)
    den = np.sum(W * Cl_theory**2)
    A   = num/den if den>0 else 0.0
    sigA= 1.0/np.sqrt(den) if den>0 else np.inf
    chi2= np.sum(W * (Cl_data - A*Cl_theory)**2)
    dof = max(len(Cl_data)-1, 1)
    return A, sigA, chi2, dof

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmap", required=True, help="Planck SMICA μK FITS")
    ap.add_argument("--gmap", required=True, help="δg FITS, or comma-separated list for tomography")
    ap.add_argument("--mask", required=True, help="0/1 or boolean FITS; same NSIDE")
    ap.add_argument("--lmax", type=int, default=100)
    ap.add_argument("--namaster", action="store_true", help="Use NaMaster (MASTER debias)")
    ap.add_argument("--nlb", type=int, default=8, help="NaMaster ℓ-bin width")
    ap.add_argument("--template-csv", default="", help="CSV with columns: ell,Cl_Tg")
    ap.add_argument("--nz-bias", default="", help="If no template CSV, recompute theory from nz,b")
    ap.add_argument("--nbar-arcmin2", type=float, default=0.5, help="Shot noise for gg error (if theory nz used)")
    ap.add_argument("--tag", default="run1", help="Label for output filenames")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    T = read_map(args.tmap)
    g_paths = [s.strip() for s in args.gmap.split(",")]
    Gs = [read_map(p) for p in g_paths]
    M = read_map(args.mask).astype(bool)

    nside = ensure_same_nside([T, M] + Gs)

    # ℓ-grid
    if args.namaster:
        ells_nmt = None  # created by NaMaster binning
    else:
        ells = np.arange(0, args.lmax+1)

    # Prepare theory if requested (single template reused for all bins unless you pass different --template-csv per bin run)
    # If you want per-bin theory with nz-splits, run script multiple times per tomographic bin with its own nz file.
    if args.template_csv or args.nz_bias:
        # build on the anafast multipole grid; if NaMaster, will be interpolated later
        the_ells = np.arange(0, args.lmax+1)
        Cl_th = load_theory_template(the_ells, args.template_csv, args.nz_bias)
    else:
        Cl_th = None

    rows_all = []
    SNcum_tot_l = None

    for i, (gmap, gpath) in enumerate(zip(Gs, g_paths)):
        label = f"bin{i+1}" if len(Gs)>1 else "all"
        if args.namaster:
            if not HAS_NM:
                raise RuntimeError("NaMaster not available; install or omit --namaster.")
            ells_eff, Cl_Tg, Cl_TT, Cl_gg, fsky = compute_cls_namaster(T, gmap, M, args.lmax, args.nlb)
            # interpolate theory to these bandpowers
            if Cl_th is not None:
                Cl_theory = np.interp(ells_eff, np.arange(0, args.lmax+1), Cl_th)
            else:
                Cl_theory = None
            ells_use = ells_eff
        else:
            Cl_Tg, Cl_TT, Cl_gg, fsky = compute_cls_anafast(T, gmap, M, args.lmax)
            ells_use = np.arange(len(Cl_Tg))
            Cl_theory = None if Cl_th is None else Cl_th.copy()

        # Keep only ℓ≥2 for ISW
        m = (ells_use>=2)
        ells_cut = ells_use[m]
        Cl_Tg_c  = Cl_Tg[m]
        Cl_TT_c  = Cl_TT[m]
        Cl_gg_c  = Cl_gg[m]

        # Variance & S/N
        sigma_l, SN_l, SN_cum = variance_and_sn(ells_cut, Cl_TT_c, Cl_gg_c, Cl_Tg_c, fsky)

        # Amplitude fit
        if Cl_theory is not None:
            Cl_th_c = Cl_theory[m]
            A, sA, chi2, dof = fit_amplitude(Cl_Tg_c, Cl_th_c, sigma_l)
        else:
            A, sA, chi2, dof = np.nan, np.nan, np.nan, max(len(ells_cut)-1,1)

        # Save CSV per bin
        csv_path = os.path.join(args.outdir, f"isw_maps_cl_tg_{args.tag}_{label}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ell","Cl_Tg","sigma_l","SN_l","SN_cum"])
            for L, c, s, s1, sC in zip(ells_cut, Cl_Tg_c, sigma_l, SN_l, SN_cum):
                w.writerow([int(L), float(c), float(s), float(s1), float(sC)])

        # Plot
        plt.figure(figsize=(7.4,5))
        plt.errorbar(ells_cut, Cl_Tg_c, yerr=sigma_l, fmt="o", ms=3, lw=1, label="data")
        if Cl_theory is not None and np.isfinite(A):
            plt.plot(ells_cut, A*Cl_th_c, label=f"theory × A={A:.2f}±{sA:.2f}")
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell^{Tg}$")
        plt.title(f"ISW cross — {label}  (f_sky={fsky:.2f})  S/N={SN_cum[-1]:.2f}")
        plt.legend(frameon=False)
        png_path = os.path.join(args.outdir, f"isw_maps_tg_plot_{args.tag}_{label}.png")
        plt.tight_layout(); plt.savefig(png_path, dpi=140); plt.close()

        # Append summary
        rows_all.append((label, fsky, SN_cum[-1], A, sA, chi2, dof, csv_path, png_path))

        # Quadrature combine S/N across bins
        if SNcum_tot_l is None:
            SNcum_tot_l = SN_l**2
        else:
            # align by ℓ; here grids match per-branch; if not, you’d interpolate
            SNcum_tot_l = SNcum_tot_l + SN_l**2

    # Combined S/N across bins (approx, assumes same ℓ grid)
    SN_tot = float(np.sqrt(np.cumsum(SNcum_tot_l)[-1])) if SNcum_tot_l is not None else 0.0

    # Write summary
    sum_path = os.path.join(args.outdir, f"isw_maps_summary_{args.tag}.txt")
    with open(sum_path, "w") as f:
        f.write("# label  f_sky  S/N_bin  A  sA  chi2  dof  csv  png\n")
        for r in rows_all:
            f.write(" ".join([str(x) for x in r])+"\n")
        f.write(f"\nTOTAL_SNR {SN_tot:.3f}\n")
    print(f"[isw_maps] wrote {sum_path}")
    for _,_,_,_,_,_,_,csvp,pngp in rows_all:
        print(f"[isw_maps] {csvp}")
        print(f"[isw_maps] {pngp}")

if __name__ == "__main__":
    main()
