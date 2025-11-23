#!/usr/bin/env python3
# Fetch a real subset of 2MPZ (RA, DEC, z_phot) from VizieR via astroquery.
# We apply |b|>25 deg and DEC>0 (halve the sky to keep file size modest).

from astroquery.vizier import Vizier
from astropy.table import vstack
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

# 1) Find the 2MPZ catalogue by name keywords
cats = Vizier.find_catalogs("2MPZ")
# Pick the first match that contains '2MPZ' in the key
catkey = next(k for k in cats.keys() if "2MPZ" in k.upper())

# 2) Configure columns & row limit (fetch up to ~400k rows to be sane)
Vizier.ROW_LIMIT = 400000
v = Vizier(columns=["RAJ2000","DEJ2000","z_phot","b"],  # 'b' = Galactic latitude
           column_filters={"b":">25", "DEJ2000":">0"})   # |b|>25, DEC>0

# 3) Wide sky query: use a coarse tiling to avoid server timeouts
# We'll query 8 centers with large radii and stack, then drop duplicates
centers = [
    SkyCoord(ra=  0*u.deg, dec=+30*u.deg, frame='icrs'),
    SkyCoord(ra= 90*u.deg, dec=+30*u.deg, frame='icrs'),
    SkyCoord(ra=180*u.deg, dec=+30*u.deg, frame='icrs'),
    SkyCoord(ra=270*u.deg, dec=+30*u.deg, frame='icrs'),
    SkyCoord(ra=  0*u.deg, dec=+75*u.deg, frame='icrs'),
    SkyCoord(ra=120*u.deg, dec=+75*u.deg, frame='icrs'),
    SkyCoord(ra=240*u.deg, dec=+75*u.deg, frame='icrs'),
    SkyCoord(ra=300*u.deg, dec=+75*u.deg, frame='icrs'),
]
rad = 70*u.deg

tables = []
for c in centers:
    r = v.query_region(c, radius=rad, catalog=catkey)
    if len(r)==0: continue
    t = r[0]
    tables.append(t)

if not tables:
    raise SystemExit("No rows returned — try re-running (VizieR can be flaky).")

T = vstack(tables, metadata_conflicts='silent')

# 4) Drop duplicates by position, keep minimal columns
T = T.to_pandas()
T = T.drop_duplicates(subset=["RAJ2000","DEJ2000"])
T = T.rename(columns={"RAJ2000":"RA","DEJ2000":"DEC","z_phot":"PHOTOZ"})
T = T[["RA","DEC","PHOTOZ"]].dropna()

# Random downsample to ~200k rows if huge
if len(T) > 200_000:
    T = T.sample(200_000, random_state=42).sort_index()

# 5) Write CSV where your pipeline expects it
import os
os.makedirs("data/lss", exist_ok=True)
out = "data/lss/2mpz_subset.csv"
T.to_csv(out, index=False)
print(f"Wrote {out} with {len(T):,} rows (|b|>25, DEC>0).")
