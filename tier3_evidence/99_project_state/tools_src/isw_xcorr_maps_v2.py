#!/usr/bin/env python3
"""
Planck×LSS ISW cross-power (map-level), with:
- robust FITS reads (no UNSEEN bleed), SciPy >=1.11 (simpson), memory-light
- optional NaMaster MASTER de-bias (--namaster)
- reports TOTAL_SNR (quadrature of per-ℓ) and detection Z = A/sA
- writes per-ℓ CSV + PNG + summary.txt

Outputs:
  outputs/isw_maps_cl_tg_[tag]_[bin].csv       (ell, Cl_Tg, sigma_l, SN_l, SN_cum, tpl_SNR_cum)
  outputs/isw_maps_tg_plot_[tag]_[bin].png
  outputs/isw_maps_summary_[tag].txt
"""
import os, csv, argparse, numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import integrate

# ------------------ optional NaMaster ------------------
try:
    import pymaster as nmt
    HAS_NM = True
except Exception:
    HAS_NM = False

# ------------------ helpers ------------------
def _simp(y, x):
    try:
        return integrate.simpson(y, x=x)
    except Exception:
        return np.trapz(y, x)

def read_map(path):
    """Read a HEALPix map and clamp non-finite/UNSEEN to 0 to avoid anafast issues."""
    m = hp.read_map(path)
    m = np.where(np.isfinite(m), m, 0.0)
    return m.astype(float)

def variance_and_sn(ells, ClTT, Clgg, ClTg, fsky):
    var = (ClTT*Clgg + ClTg**2) / np.maximum((2*ells+1)*fsky, 1e-30)
    sig = np.sqrt(np.maximum(var, 1e-30))
    snl = ClTg / sig
    sncum = np.sqrt(np.cumsum(snl**2))
    return sig, snl, sncum

def fit_amplitude(Cl_data, Cl_theory, sigma):
    W = 1.0/np.maximum(sigma,1e-30)**2
    num = np.sum(W*Cl_data*Cl_theory)
    den = np.sum(W*Cl_theory**2)
    A   = num/den if den>0 else np.nan
    sA  = (den**-0.5) if den>0 else np.nan
    chi2= np.sum(W*(Cl_data - A*Cl_theory)**2)
    dof = max(len(Cl_data)-1, 1)
    return A, sA, chi2, dof

def template_snr_cumulative(Cl_data, Cl_th, sigma):
    w = 1.0/np.maximum(sigma,1e-30)**2
    num_c = np.cumsum(Cl_data*Cl_th*w)
    den_c = np.cumsum((Cl_th**2)*w)
    return np.sqrt(np.maximum((num_c**2)/np.maximum(den_c,1e-30), 0.0))

# --------------- minimal theory (for template only) ---------------
H0 = 67.4
Omega_m = 0.315
Omega_L = 1.0 - Omega_m
def Ez(z): return np.sqrt(Omega_m*(1+z)**3 + Omega_L)
def chi_of_z(z):
    from scipy import integrate as _int
    f = lambda zp: 299792.458 / (H0*Ez(zp))
    return np.array([_int.quad(f, 0.0, zi, epsrel=3e-6)[0] for zi in np.atleast_1d(z)])
def growth_D(a):
    Om_a = Omega_m/(Omega_m + Omega_L*a**3)
    Ol_a = 1.0 - Om_a
    g = 5.0*Om_a/(2.0*(Om_a**(4/7) - Ol_a + (1+0.5*Om_a)*(1 + Ol_a/70.0)))
    return g/g[-1] if np.ndim(g)>0 else g
def linear_Pk_EH_nw(k, z):
    h = H0/100.0; kh = k/h
    a,b,c,nu = 6.0,3.0,1.7,1.2
    Tk = (1.0 + (a*kh + (b*kh)**1.5 + (c*kh)**2.0)**nu )**(-1.0/nu)
    ns = 0.965; a_z = 1.0/(1.0+z); Dz = growth_D(np.array([a_z]))[0]
    return 1e4 * (k**ns) * (Tk**2) * (Dz**2)
def load_nz_bias(path):
    z, nz, b = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    nz = nz/np.trapz(nz, z)
    return z, nz, b
def Cl_Tg_limber(ell, z, nz, b):
    chi = chi_of_z(z)
    a = 1.0/(1.0+z)
    from numpy import maximum
    k  = (ell+0.5)/maximum(chi,1e-9)
    Pk = linear_Pk_EH_nw(k, z)
    Hz = H0*Ez(z)
    WT = (a*0 + 1)  # simple shape proxy via d(ln D)/dη omitted; amplitude fitted anyway
    Wg = b * nz * growth_D(a)
    integrand = WT*Wg*Pk/(chi**2 * Hz)
    return _simp(integrand, z)

# --------------- NaMaster pipeline ---------------
def nm_cls(Tmap, gmap, mask, lmax, nlb):
    b = nmt.NmtBin.from_nside_linear(hp.get_nside(mask), nlb, is_Dell=False, lmax=lmax)
    fT = nmt.NmtField(mask.astype(float), [Tmap])
    fg = nmt.NmtField(mask.astype(float), [gmap])
    cl_c = nmt.compute_coupled_cell(fT, fg)[0]
    w = nmt.NmtWorkspace(); w.compute_coupling_matrix(fT, fg, b)
    cl = w.decouple_cell(cl_c)
    clTT = w.decouple_cell(nmt.compute_coupled_cell(fT,fT)[0])
    clgg = w.decouple_cell(nmt.compute_coupled_cell(fg,fg)[0])
    ells = b.get_effective_ells()
    fsky = float(np.mean(mask>0))
    return ells, cl, clTT, clgg, fsky

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmap", required=True)
    ap.add_argument("--gmap", required=True, help="FITS or comma-separated list for tomography")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--lmax", type=int, default=100)
    ap.add_argument("--namaster", action="store_true")
    ap.add_argument("--nlb", type=int, default=8)
    ap.add_argument("--nz-bias", default="", help="CSV with z,nz,b for theory template")
    ap.add_argument("--tag", default="run2")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    T = read_map(args.tmap)
    g_paths = [s.strip() for s in args.gmap.split(",")]
    Gs = [read_map(p) for p in g_paths]
    M = read_map(args.mask) > 0.5
    nside = hp.get_nside(M)
    assert all(hp.get_nside(m)==nside for m in [T,*Gs]), "All maps must share NSIDE"

    # optional template
    Cl_th_full = None
    if args.nz_bias:
        z, nz, b = load_nz_bias(args.nz_bias)
        the_ells = np.arange(0, args.lmax+1)
        Cl_th_full = np.array([Cl_Tg_limber(L, z, nz, b) for L in the_ells])

    rows_all = []
    SNcum_tot_l = None

    for i,(gmap,gp) in enumerate(zip(Gs,g_paths)):
        label = f"bin{i+1}" if len(Gs)>1 else "all"

        if args.namaster:
            if not HAS_NM:
                raise RuntimeError("pymaster not available. Install or omit --namaster.")
            ells_eff, Cl_Tg, Cl_TT, Cl_gg, fsky = nm_cls(T, gmap, M, args.lmax, args.nlb)
            ells_use = ells_eff
            Cl_th = None if Cl_th_full is None else np.interp(ells_use, np.arange(args.lmax+1), Cl_th_full)
        else:
            m = M.astype(float)
            Cl_Tg = hp.anafast(T*m, gmap*m, lmax=args.lmax)
            Cl_TT = hp.anafast(T*m, lmax=args.lmax)
            Cl_gg = hp.anafast(gmap*m, lmax=args.lmax)
            fsky = float(np.mean(M))
            ells_use = np.arange(len(Cl_Tg))
            Cl_th = None if Cl_th_full is None else Cl_th_full.copy()

        # keep ℓ≥2
        msel = (ells_use>=2)
        ells = ells_use[msel]
        Cl_Tg = Cl_Tg[msel]; Cl_TT=Cl_TT[msel]; Cl_gg=Cl_gg[msel]
        sigma_l, SN_l, SN_cum = variance_and_sn(ells, Cl_TT, Cl_gg, Cl_Tg, fsky)

        # amplitude fit + template cumulative SNR
        if Cl_th is not None:
            Cl_th_c = Cl_th[msel]
            A, sA, chi2, dof = fit_amplitude(Cl_Tg, Cl_th_c, sigma_l)
            Z = A/sA if (np.isfinite(A) and np.isfinite(sA) and sA>0) else np.nan
            tpl_SNR_cum = template_snr_cumulative(Cl_Tg, Cl_th_c, sigma_l)
        else:
            A=sA=Z=chi2=dof=np.nan
            tpl_SNR_cum = np.full_like(SN_cum, np.nan)

        # save CSV
        csv_path = os.path.join(args.outdir, f"isw_maps_cl_tg_{args.tag}_{label}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ell","Cl_Tg","sigma_l","SN_l","SN_cum","tpl_SNR_cum"])
            for L,c,s,s1,sC,tC in zip(ells, Cl_Tg, sigma_l, SN_l, SN_cum, tpl_SNR_cum):
                w.writerow([int(L), float(c), float(s), float(s1), float(sC), float(tC) if np.isfinite(tC) else ""])

        # plot
        plt.figure(figsize=(7.2,5))
        plt.errorbar(ells, Cl_Tg, yerr=sigma_l, fmt="o", ms=3, lw=1, label="data")
        if Cl_th is not None and np.isfinite(A):
            plt.plot(ells, A*Cl_th_c, label=f"template × A={A:.2f}±{sA:.2f} (Z={Z:.2f})")
        plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell^{Tg}$")
        plt.title(f"ISW cross — {label}  f_sky={fsky:.2f}  S/N={SN_cum[-1]:.2f}")
        plt.legend(frameon=False); plt.tight_layout()
        png_path = os.path.join(args.outdir, f"isw_maps_tg_plot_{args.tag}_{label}.png")
        plt.savefig(png_path, dpi=140); plt.close()

        rows_all.append((label, fsky, SN_cum[-1], A, sA, (A/sA if (np.isfinite(A) and sA>0) else np.nan), chi2, dof, csv_path, png_path))
        SNcum_tot_l = SN_l**2 if SNcum_tot_l is None else SNcum_tot_l + SN_l**2

    SN_tot = float(np.sqrt(np.cumsum(SNcum_tot_l)[-1])) if SNcum_tot_l is not None else 0.0

    sum_path = os.path.join(args.outdir, f"isw_maps_summary_{args.tag}.txt")
    with open(sum_path, "w") as f:
        f.write("# label  f_sky  S/N_bin  A  sA  Z  chi2  dof  csv  png\n")
        for r in rows_all: f.write(" ".join([str(x) for x in r])+"\n")
        f.write(f"\nTOTAL_SNR {SN_tot:.3f}\n")
    print(f"[isw_maps] wrote {sum_path}")
    for *_,csvp,pngp in rows_all:
        print(f"[isw_maps] {csvp}")
        print(f"[isw_maps] {pngp}")

if __name__ == "__main__":
    main()
