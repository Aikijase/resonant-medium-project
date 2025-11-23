#!/usr/bin/env python3
"""
Parse DESI DR1 Gaussian BAO mean/cov files into a single CSV your validator accepts.

Output: data/bao_aniso_desidr1.csv
Columns supported per row:
  z,DM_over_rd,err_DM_over_rd,DH_over_rd,err_DH_over_rd,DV_over_rd,err_DV_over_rd

Notes
- BGS and QSO are isotropic (DV/rd only in DR1 Gaussian format).
- LRG1, LRG2, LRG+ELG, ELG have anisotropic (DM/rd, DH/rd).
- Lyα is optional (anisotropic too), can be appended if desired.
"""

import csv, numpy as np
from pathlib import Path

DATA = Path("data/desi_dr1_bao")
OUT  = Path("data/bao_aniso_desidr1.csv")

# Map tracer -> (mean_file, cov_file, expected order of quantities in mean vector)
# The Gaussian 'mean' files have lines like: "<z> <value> <name>"
CFG = {
    "BGS":    ("BGS_mean.txt",    "BGS_cov.txt",    ["DV_over_rs"]),
    "LRG1":   ("LRG1_mean.txt",   "LRG1_cov.txt",   ["DM_over_rs","DH_over_rs"]),
    "LRG2":   ("LRG2_mean.txt",   "LRG2_cov.txt",   ["DM_over_rs","DH_over_rs"]),
    "LRGELG": ("LRGELG_mean.txt", "LRGELG_cov.txt", ["DM_over_rs","DH_over_rs"]),
    "ELG":    ("ELG_mean.txt",    "ELG_cov.txt",    ["DM_over_rs","DH_over_rs"]),
    "QSO":    ("QSO_mean.txt",    "QSO_cov.txt",    ["DV_over_rs"]),
    # "LYA":  ("LYA_mean.txt",     "LYA_cov.txt",    ["DM_over_rs","DH_over_rs"]),
}

def read_mean(path):
    z = None
    vals = {}
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if len(parts) < 3: continue
        zi, vi, qi = parts[0], parts[1], parts[2]
        try:
            zf = float(zi); vf = float(vi)
        except:
            continue
        z = zf  # they all have the same z per file
        vals[qi] = vf
    return z, vals

def read_cov(path, names):
    """
    Cov files are square numeric matrices with the same ordering as the 'names' list.
    We read all numbers and reshape to NxN.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        # accept both space- or comma-separated numeric rows
        try:
            row = [float(x) for x in parts]
            rows.append(row)
        except:
            continue
    mat = np.array(rows, dtype=float)
    # Some files are 1x1; others 2x2. If they came in flattened, try to reshape.
    if mat.ndim == 2 and mat.shape[0] != mat.shape[1] and mat.size == len(names)**2:
        mat = mat.reshape((len(names), len(names)))
    if mat.ndim == 1 and mat.size == len(names)**2:
        mat = mat.reshape((len(names), len(names)))
    if mat.shape != (len(names), len(names)):
        raise ValueError(f"Covariance shape {mat.shape} does not match {len(names)} names.")
    return mat

rows_out = []
for key,(mfile,cfile,names) in CFG.items():
    mf = DATA/mfile; cf = DATA/cfile
    if not mf.exists():
        print(f"WARNING: missing {mf}, skipping {key}")
        continue
    z, vals = read_mean(mf)
    # sanity: ensure we have the requested quantities
    for q in names:
        if q not in vals:
            raise ValueError(f"{key}: quantity {q} not found in {mf}")
    cov = None
    if cf.exists():
        try:
            cov = read_cov(cf, names)
        except Exception as e:
            print(f"WARNING: could not parse cov for {key}: {e}")

    # build output row depending on what exists
    row = {"z": f"{z:.3f}"}
    if names == ["DV_over_rs"]:
        dv = vals["DV_over_rs"]
        row["DV_over_rd"] = f"{dv:.6f}"
        if cov is not None:
            err = np.sqrt(cov[0,0])
            row["err_DV_over_rd"] = f"{err:.6f}"
    else:
        dm = vals["DM_over_rs"]; dh = vals["DH_over_rs"]
        row["DM_over_rd"] = f"{dm:.6f}"
        row["DH_over_rd"] = f"{dh:.6f}"
        if cov is not None:
            sdm = np.sqrt(cov[0,0]); sdh = np.sqrt(cov[1,1])
            row["err_DM_over_rd"] = f"{sdm:.6f}"
            row["err_DH_over_rd"] = f"{sdh:.6f}"

    rows_out.append(row)

# write CSV with union of columns present
cols = ["z","DM_over_rd","err_DM_over_rd","DH_over_rd","err_DH_over_rd","DV_over_rd","err_DV_over_rd"]
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows_out:
        w.writerow({k:r.get(k,"") for k in cols})

print(f"Wrote {OUT}")
