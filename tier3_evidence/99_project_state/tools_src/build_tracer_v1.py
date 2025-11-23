#!/usr/bin/env python3
"""
Build δg HEALPix maps (and nz_bias.csv) from a catalog.

Features
- Input: CSV or FITS with RA, DEC (deg). Optional redshift column.
- Auto-detect RA/DEC column names (overridable).
- Optional tomography: --z-bins 0.0,0.3,0.6,1.0  → writes one δg per bin.
- Writes:
    data/lss/delta_g_nside{NSIDE}.fits                  (if no z-bins)
    data/lss/delta_g_bin{i}_nside{NSIDE}.fits          (if z-bins)
    data/nz_bias.csv                                    (from catalog; normalized)
- Mask: any FITS map (True inside footprint); default uses your Planck |b|>25 mask.
- Bias model: python expr in z, e.g. "1+0.8*z" (default), or constant like "1.4".
- Lightweight: clamps outside-mask to UNSEEN only at the end; uses float32 where sensible.

Usage (examples at bottom).
"""
import os, argparse, numpy as np
import healpy as hp

def infer_cols(names):
    ra_cands  = ["ra","raj2000","alpha_j2000","ra_deg","radegree","ra_icrs","alpha","radeg"]
    de_cands  = ["dec","dej2000","delta_j2000","dec_deg","decl","dec_icrs","delta","decdeg"]
    low = [n.lower() for n in names]
    ra = next((names[i] for i,s in enumerate(low) if s in ra_cands), None)
    de = next((names[i] for i,s in enumerate(low) if s in de_cands), None)
    return ra, de

def read_catalog(path, fmt, ra_col=None, dec_col=None, z_col=None, max_rows=None):
    if fmt == "csv":
        import pandas as pd
        # first pass to infer columns if not provided
        head = pd.read_csv(path, nrows=1)
        if not ra_col or not dec_col:
            ra_col_inf, de_col_inf = infer_cols(head.columns)
            ra_col = ra_col or ra_col_inf
            dec_col = dec_col or de_col_inf
            if not (ra_col and dec_col):
                raise ValueError(f"Could not infer RA/DEC columns in CSV. Have: {list(head.columns)} "
                                 f"Use --ra-col and --dec-col.")
        usecols = [ra_col, dec_col] + ([z_col] if z_col else [])
        df = pd.read_csv(path, usecols=usecols) if max_rows is None else pd.read_csv(path, usecols=usecols, nrows=max_rows)
        ra  = df[ra_col].to_numpy(float)
        dec = df[dec_col].to_numpy(float)
        z   = df[z_col].to_numpy(float) if z_col else None
        m = np.isfinite(ra) & np.isfinite(dec) & (np.isfinite(z) if z_col else True)
        return ra[m], dec[m], (z[m] if z_col else None)
    elif fmt == "fits":
        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            tab = hdul[1].data
            names = list(hdul[1].columns.names)
            if not ra_col or not dec_col:
                ra_col_inf, de_col_inf = infer_cols(names)
                ra_col = ra_col or ra_col_inf
                dec_col = dec_col or de_col_inf
                if not (ra_col and dec_col):
                    raise ValueError(f"Could not infer RA/DEC columns in FITS. Have: {names} "
                                     f"Use --ra-col and --dec-col.")
            ra = np.asarray(tab[ra_col], dtype=float)
            dec= np.asarray(tab[dec_col], dtype=float)
            z  = np.asarray(tab[z_col], dtype=float) if z_col else None
            m = np.isfinite(ra) & np.isfinite(dec) & (np.isfinite(z) if z_col else True)
            return ra[m], dec[m], (z[m] if z_col else None)
    else:
        raise ValueError("fmt must be csv or fits")

def make_delta_from_radec(ra, dec, mask_bool, nside):
    theta = np.radians(90.0 - dec)
    phi   = np.radians(ra)
    pix   = hp.ang2pix(nside, theta, phi)
    npix  = hp.nside2npix(nside)
    counts= np.bincount(pix, minlength=npix).astype(np.float64)
    idx = np.where(mask_bool)[0]
    mu  = counts[idx].mean() if idx.size>0 else 0.0
    delta = np.full(npix, hp.UNSEEN, float)
    if idx.size>0 and mu>0:
        delta[idx] = (counts[idx]-mu)/(mu+1e-12)
    else:
        delta[idx] = 0.0
    return delta

def write_nz_bias_from_z(z, out_csv, bias_expr="1+0.8*z", zmin=None, zmax=None, nbins=60):
    if z is None or z.size==0:
        raise ValueError("No redshifts provided to build nz_bias.csv")
    if zmin is None: zmin = max(1e-3, np.percentile(z, 0.5))
    if zmax is None: zmax = np.percentile(z, 99.5)
    bins = np.linspace(zmin, zmax, nbins+1)
    zc   = 0.5*(bins[:-1]+bins[1:])
    hist,_ = np.histogram(z, bins=bins)
    nz = hist.astype(float)
    # enforce nonnegativity and normalize
    nz = np.maximum(nz, 0.0)
    if nz.sum() == 0:
        nz[:] = 1.0
    nz = nz / np.trapz(nz, zc)
    # bias from expression
    local = {}
    b = eval(str(bias_expr), {"__builtins__": {}}, {"z": zc, "np": np})
    b = np.asarray(b, dtype=float)
    import pandas as pd, os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame({"z":zc, "nz":nz, "b":b}).to_csv(out_csv, index=False)
    return zc, nz, b

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cat", required=False, default="", help="Path to catalog (CSV or FITS). If omitted and none found, a synthetic δg is made.")
    ap.add_argument("--fmt", choices=["csv","fits"], required=False, help="Catalog format. If omitted, inferred from extension.")
    ap.add_argument("--ra-col", default="", help="RA column name (deg). If omitted, try to infer.")
    ap.add_argument("--dec-col", default="", help="DEC column name (deg). If omitted, try to infer.")
    ap.add_argument("--z-col", default="", help="Redshift column (optional).")
    ap.add_argument("--z-bins", default="", help="Comma list of edges for tomography, e.g. 0.0,0.3,0.6,1.0")
    ap.add_argument("--nside", type=int, default=128)
    ap.add_argument("--mask", default="data/planck/mask_gal25_nside128.fits", help="Footprint mask (True inside).")
    ap.add_argument("--outdir", default="data/lss")
    ap.add_argument("--name", default="tracer", help="Base name for outputs.")
    ap.add_argument("--bias-expr", default="1+0.8*z", help="Bias model for nz_bias.csv, python expr in z (e.g. '1.2+0.6*z').")
    ap.add_argument("--max-rows", type=int, default=None, help="Read only first N rows (debug).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mask = hp.read_map(args.mask)
    M = np.asarray(mask>0.5, dtype=bool)
    nside = args.nside
    if hp.get_nside(mask) != nside:
        # downgrade/upgrade conservatively to the requested nside
        M = hp.ud_grade(M.astype(float), nside_out=nside, pess=True) > 0.99

    # Locate catalog if not specified
    cat = args.cat
    fmt = args.fmt
    if not cat:
        import glob
        cands = sorted(glob.glob("data/lss/*.csv")+glob.glob("data/lss/**/*.csv", recursive=True)
                       +glob.glob("data/lss/*.fits")+glob.glob("data/lss/**/*.fits", recursive=True))
        if cands:
            cat = cands[0]
            print(f"[info] Using catalog: {cat}")
        else:
            print("[info] No catalog found. Creating synthetic δg.")
            delta = make_delta_from_radec(np.array([]), np.array([]), M, nside)  # returns zeros in-mask
            hp.write_map(os.path.join(args.outdir, f"delta_g_nside{nside}.fits"), delta, overwrite=True)
            print("Wrote", os.path.join(args.outdir, f"delta_g_nside{nside}.fits"))
            return

    if not fmt:
        ext = os.path.splitext(cat)[1].lower()
        fmt = "csv" if ext==".csv" else ("fits" if ext==".fits" else None)
        if not fmt:
            raise ValueError("Cannot infer format from extension; pass --fmt csv|fits")

    ra, dec, z = read_catalog(cat, fmt, args.ra_col or None, args.dec_col or None, args.z_col or None, args.max_rows)
    print(f"[info] Loaded {ra.size} objects (z present: {z is not None})")

    # Tomography?
    if args.z_bins and not z is None:
        edges = np.array([float(x) for x in args.z_bins.split(",")], dtype=float)
        if edges.ndim!=1 or edges.size<2:
            raise ValueError("Need at least two edges for --z-bins")
        for i in range(len(edges)-1):
            zmin, zmax = edges[i], edges[i+1]
            sel = (z>=zmin) & (z<zmax)
            if sel.sum()==0:
                print(f"[warn] Bin {i+1} ({zmin},{zmax}) is empty; writing UNSEEN map.")
            delta = make_delta_from_radec(ra[sel], dec[sel], M, nside)
            out = os.path.join(args.outdir, f"delta_g_bin{i+1}_nside{nside}.fits")
            hp.write_map(out, delta, overwrite=True)
            print("Wrote", out)
        # nz_bias for the whole tracer (not per bin) based on supplied bias-expr
        try:
            zc, nzv, bv = write_nz_bias_from_z(z, "data/nz_bias.csv", args.bias_expr)
            print("Wrote data/nz_bias.csv  (from catalog z)")
        except Exception as e:
            print("[warn] nz_bias.csv not written:", e)
    else:
        delta = make_delta_from_radec(ra, dec, M, nside)
        out = os.path.join(args.outdir, f"delta_g_nside{nside}.fits")
        hp.write_map(out, delta, overwrite=True)
        print("Wrote", out)
        # nz_bias if z available
        if z is not None and z.size>0:
            try:
                zc, nzv, bv = write_nz_bias_from_z(z, "data/nz_bias.csv", args.bias_expr)
                print("Wrote data/nz_bias.csv  (from catalog z)")
            except Exception as e:
                print("[warn] nz_bias.csv not written:", e)

if __name__ == "__main__":
    main()
