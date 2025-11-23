#!/usr/bin/env python3
"""
Build a HEALPix overdensity map δg from a catalog (CSV or FITS) under data/lss/.
- auto-detects catalog & RA/DEC columns; if none found, makes a synthetic δg
- outputs: data/lss/delta_g_nside{NSIDE}.fits
"""
import os, glob, numpy as np, healpy as hp

NSIDE = 128
MASK  = "data/planck/mask_gal25_nside128.fits"

def detect_cols(names):
    RA_CANDS  = ["ra","raj2000","alpha_j2000","ra_deg","radegree","ra_icrs","alpha"]
    DEC_CANDS = ["dec","dej2000","delta_j2000","dec_deg","decl","dec_icrs","delta"]
    low = [n.lower() for n in names]
    ra = next((names[i] for i,s in enumerate(low) if s in RA_CANDS), None)
    de = next((names[i] for i,s in enumerate(low) if s in DEC_CANDS), None)
    return ra, de

def try_csv(path):
    import pandas as pd
    sample = pd.read_csv(path, nrows=1)
    ra_c, de_c = detect_cols(sample.columns)
    if not (ra_c and de_c):
        raise ValueError(f"No RA/DEC columns detected in {path}. Have: {list(sample.columns)}")
    df = pd.read_csv(path, usecols=[ra_c,de_c]).dropna()
    return df[ra_c].to_numpy(float), df[de_c].to_numpy(float), {"file": path, "RAcol": ra_c, "DECcol": de_c}

def try_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        names = list(hdul[1].columns.names)
        ra_c, de_c = detect_cols(names)
        if not (ra_c and de_c):
            raise ValueError(f"No RA/DEC columns detected in {path}. Have: {names}")
        tab = hdul[1].data
        ra  = np.asarray(tab[ra_c], dtype=float)
        dec = np.asarray(tab[de_c], dtype=float)
        m = np.isfinite(ra) & np.isfinite(dec)
        return ra[m], dec[m], {"file": path, "RAcol": ra_c, "DECcol": de_c}

def main():
    os.makedirs("data/lss", exist_ok=True)
    mask = hp.read_map(MASK).astype(bool)

    csvs = sorted(glob.glob("data/lss/**/*.csv", recursive=True) + glob.glob("data/lss/*.csv"))
    fitsf= sorted(glob.glob("data/lss/**/*.fits", recursive=True) + glob.glob("data/lss/*.fits"))

    ra = dec = info = None
    if csvs:
        try:
            ra, dec, info = try_csv(csvs[0])
        except Exception as e:
            print("[warn] CSV failed:", e)
    if info is None and fitsf:
        try:
            ra, dec, info = try_fits(fitsf[0])
        except Exception as e:
            print("[warn] FITS failed:", e)

    if info is None:
        print("[info] No catalog found; creating synthetic δg.")
        npix = hp.nside2npix(NSIDE)
        idx = np.where(mask)[0]
        lam = 30.0
        counts = np.zeros(npix); counts[idx] = np.random.poisson(lam, size=len(idx))
        mu = counts[idx].mean()
        delta = np.full(npix, hp.UNSEEN); delta[idx] = (counts[idx]-mu)/(mu+1e-12)
    else:
        print(f"[info] Using catalog: {info['file']} (RA={info['RAcol']} DEC={info['DECcol']})")
        theta = np.radians(90.0 - dec); phi = np.radians(ra)
        pix = hp.ang2pix(NSIDE, theta, phi)
        npix = hp.nside2npix(NSIDE)
        counts = np.bincount(pix, minlength=npix).astype(float)
        idx = np.where(mask)[0]
        mu = counts[idx].mean()
        delta = np.full(npix, hp.UNSEEN)
        delta[idx] = (counts[idx]-mu)/(mu+1e-12)

    out = f"data/lss/delta_g_nside{NSIDE}.fits"
    hp.write_map(out, delta, overwrite=True)
    print("Wrote", out)

if __name__ == "__main__":
    main()
