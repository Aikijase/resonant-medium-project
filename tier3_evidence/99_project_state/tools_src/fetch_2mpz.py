#!/usr/bin/env python3
"""
Fetch a real subset of 2MPZ (J/ApJS/210/9) with RA, DEC, PHOTOZ via VizieR.

Strategy:
1) Try VizieR by explicit catalog id 'J/ApJS/210/9' and scan tables to find a photo-z column.
2) If that fails, query TAPVizieR with ADQL against J/ApJS/210/9 and pick likely z_phot/photoz columns.
3) Filter to |b|>25 (to match your Planck mask) and DEC>0 to keep file size modest.
4) Write data/lss/2mpz_phot.csv with columns: RA, DEC, PHOTOZ
"""

import os, sys, time
import numpy as np
import pandas as pd

# ensure output dir
os.makedirs("data/lss", exist_ok=True)
OUT = "data/lss/2mpz_phot.csv"

def write_and_exit(df):
    df = df[['RA','DEC','PHOTOZ']].dropna()
    # restrict to |b|>25 and DEC>0
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    b = SkyCoord(ra=df['RA'].values*u.deg, dec=df['DEC'].values*u.deg).galactic.b.deg
    df = df[(np.abs(b)>25) & (df['DEC']>0)]
    # downsample if huge
    if len(df) > 400_000:
        df = df.sample(400_000, random_state=42)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} rows={len(df)} cols={list(df.columns)}")
    sys.exit(0)

def try_vizier_catalogs():
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1  # we'll limit later
    # Pull all tables in the catalog
    cats = Vizier.get_catalogs("J/ApJS/210/9")
    for t in cats:
        df = t.to_pandas()
        cols = {c.lower(): c for c in df.columns}
        # try to detect RA/DEC (usually RAJ2000/DEJ2000) and photo-z
        ra = cols.get("raj2000") or cols.get("radeg") or cols.get("ra")
        de = cols.get("dej2000") or cols.get("dedeg") or cols.get("dec")
        zc = (cols.get("z_phot") or cols.get("zphot") or
              cols.get("photoz") or cols.get("zbest") or
              next((c for c in df.columns if ('z' in c.lower() and 'phot' in c.lower())), None))
        if ra and de and zc:
            sub = df[[ra, de, zc]].rename(columns={ra:'RA', de:'DEC', zc:'PHOTOZ'})
            # numeric coercion
            sub = sub.apply(pd.to_numeric, errors='coerce').dropna()
            # limit size for safety
            if len(sub) > 800_000:
                sub = sub.sample(800_000, random_state=42)
            return sub
    return None

def try_vizier_region_tiles():
    # Query by large tiles to avoid server-side row limits
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    Vizier.ROW_LIMIT = 200000  # per tile
    v = Vizier(columns=["RAJ2000","DEJ2000","z_phot","zphot","photoz"])
    centers = [
        SkyCoord(0*u.deg,   20*u.deg, frame='icrs'),
        SkyCoord(120*u.deg, 20*u.deg, frame='icrs'),
        SkyCoord(240*u.deg, 20*u.deg, frame='icrs'),
        SkyCoord(60*u.deg,  60*u.deg, frame='icrs'),
        SkyCoord(180*u.deg, 60*u.deg, frame='icrs'),
        SkyCoord(300*u.deg, 60*u.deg, frame='icrs'),
    ]
    rad = 60*u.deg
    tabs = []
    for c in centers:
        r = v.query_region(c, radius=rad, catalog="J/ApJS/210/9")
        if len(r)==0: 
            continue
        tabs.append(r[0].to_pandas())
        time.sleep(0.5)
    if not tabs:
        return None
    df = pd.concat(tabs, ignore_index=True).drop_duplicates()
    cols = {c.lower(): c for c in df.columns}
    ra = cols.get("raj2000") or cols.get("radeg") or cols.get("ra")
    de = cols.get("dej2000") or cols.get("dedeg") or cols.get("dec")
    zc = (cols.get("z_phot") or cols.get("zphot") or
          cols.get("photoz") or cols.get("zbest") or
          next((c for c in df.columns if ('z' in c.lower() and 'phot' in c.lower())), None))
    if ra and de and zc:
        return df[[ra, de, zc]].rename(columns={ra:'RA', de:'DEC', zc:'PHOTOZ'})
    return None

def try_tap_adql():
    # TAP direct ADQL against TAPVizieR; keep selection small and explicit
    from astroquery.utils.tap.core import TapPlus
    ep = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
    tap = TapPlus(url=ep)
    # The main table is often "J/ApJS/210/9/table2". Select only needed cols and limit rows.
    adql = """
    SELECT TOP 800000 RAJ2000, DEJ2000,
           (CASE WHEN (defined(z_phot)) THEN z_phot
                 WHEN (defined(zphot))  THEN zphot
                 WHEN (defined(photoz)) THEN photoz
            ELSE NULL END) AS zph
    FROM "J/ApJS/210/9/table2"
    WHERE (z_phot IS NOT NULL OR zphot IS NOT NULL OR photoz IS NOT NULL)
    """
    job = tap.launch_job(adql, dump_to_file=False)
    res = job.get_results()
    if res is None or len(res)==0:
        return None
    df = res.to_pandas().rename(columns={'raj2000':'RA','dej2000':'DEC','zph':'PHOTOZ'})
    return df

# Try methods in order, with gentle retries
for fn in (try_vizier_catalogs, try_vizier_region_tiles, try_tap_adql):
    for _ in range(2):
        try:
            df = fn()
            if df is not None and len(df) > 0:
                write_and_exit(df)
        except Exception as e:
            print(f"[warn] {fn.__name__} failed: {e}")
            time.sleep(1)

print("ERROR: Could not fetch 2MPZ via VizieR/TAP right now. Try again later or with a different network.")
sys.exit(2)
