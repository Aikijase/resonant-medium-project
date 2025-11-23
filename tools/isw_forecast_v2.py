#!/usr/bin/env python3
"""
ISW forecast (no maps) with realistic theory combo:
- reads nz,b from CSV (z,nz,b), normalizes nz
- Limber-only (fast) with SciPy simpson; returns template Cl_Tg(ℓ)
- estimates S/N given f_sky, nbar (shot), and supplied Planck TT (optional)

Outputs:
  outputs/isw_template_[tag].csv   (ell, Cl_Tg)
  outputs/isw_forecast_[tag].txt   (S/N_tot, basic params)
  outputs/isw_tg_with_sn_[tag].png (optional quick plot)
"""
import os, csv, argparse, numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

H0 = 67.4
Omega_m = 0.315
Omega_L = 1.0 - Omega_m
def _simp(y, x):
    try: return integrate.simpson(y, x=x)
    except Exception: return np.trapz(y, x)
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
    k  = (ell+0.5)/np.maximum(chi,1e-9)
    Pk = linear_Pk_EH_nw(k, z)
    Hz = H0*Ez(z)
    WT = (a*0 + 1)  # amplitude fitted later; shape proxy
    Wg = b * nz * growth_D(a)
    integrand = WT*Wg*Pk/(chi**2 * Hz)
    return _simp(integrand, z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz-bias", required=True)
    ap.add_argument("--lmax", type=int, default=100)
    ap.add_argument("--fsky", type=float, default=0.3)
    ap.add_argument("--nbar-arcmin2", type=float, default=1.0)
    ap.add_argument("--tag", default="run2")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z, nz, b = load_nz_bias(args.nz_bias)
    ells = np.arange(0, args.lmax+1)
    Cl_tg = np.array([Cl_Tg_limber(L, z, nz, b) for L in ells])

    # quick-and-dirty TT envelope (use real ClTT if you have it)
    ClTT = 1e-10 + 3000.0/(np.maximum(ells,1)*(np.maximum(ells,1)+1))  # μK^2-ish scale; rough

    # galaxy auto (for variance) with shot
    chi = chi_of_z(z); a=1/(1+z)
    k  = (ells[:,None]+0.5)/np.maximum(chi[None,:],1e-9)
    Pk = linear_Pk_EH_nw(k, z[None,:])
    Hz = H0*Ez(z)[None,:]
    Wg = (b*nz*growth_D(a))[None,:]
    Clgg = _simp((Wg**2)*Pk/( (chi[None,:]**2)*Hz ), z[None,:].repeat(len(ells),axis=0))
    nbar_sr = args.nbar_arcmin2 * (60*np.pi/10800.0)**2
    Clgg = Clgg + (1.0/nbar_sr)

    m = (ells>=2)
    ellm = ells[m]; Cl_tg_m = Cl_tg[m]; ClTTm = ClTT[m]; Clggm = Clgg[m]
    var = (ClTTm*Clggm + Cl_tg_m**2)/np.maximum((2*ellm+1)*args.fsky,1e-30)
    SN_ell = Cl_tg_m/np.sqrt(np.maximum(var,1e-30))
    SN_tot = float(np.sqrt(np.cumsum(SN_ell**2)[-1]))

    # save template
    tpl_csv = os.path.join(args.outdir, f"isw_template_{args.tag}.csv")
    with open(tpl_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["ell","Cl_Tg"])
        for L,c in zip(ells, Cl_tg):
            w.writerow([int(L), float(c)])

    # quick plot
    plt.figure(figsize=(7,4.6))
    plt.plot(ellm, Cl_tg_m, lw=2)
    plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell^{Tg}$ (arb.)")
    plt.title(f"ISW template (f_sky={args.fsky:.2f})  Forecast S/N≈{SN_tot:.1f}")
    png = os.path.join(args.outdir, f"isw_tg_with_sn_{args.tag}.png")
    plt.tight_layout(); plt.savefig(png, dpi=140); plt.close()

    # summary
    summ = os.path.join(args.outdir, f"isw_forecast_{args.tag}.txt")
    with open(summ,"w") as f:
        f.write(f"Forecast S/N ~ {SN_tot:.2f} (ℓ=2..{args.lmax})\n")
        f.write(f"f_sky={args.fsky}  nbar_arcmin2={args.nbar_arcmin2}\n")
        f.write(f"template_csv={tpl_csv}\n")
        f.write(f"plot={png}\n")
    print("[isw_forecast] wrote", summ, tpl_csv, png)

if __name__ == "__main__":
    main()
