#!/usr/bin/env python3
"""
bao_fid_generator.py
Create a fiducial BAO distance table with DM/rd and DH/rd at chosen redshifts.
"""

import argparse, csv, math, sys
from pathlib import Path
import numpy as np

C_KMS = 299792.458  # km/s

def ensure_dir_for(path: str):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)

def write_fid(path, rows):
    ensure_dir_for(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z","DM_over_rd_fid","DH_over_rd_fid"])
        w.writerows(rows)
    print(f"Wrote {path}")

def read_table(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r); cols = r.fieldnames
    return rows, cols

def E2_LCDM(om0):
    ode0 = 1.0 - om0
    return lambda a: om0*a**(-3) + ode0

def build_E2_from_table(path):
    rows, cols = read_table(path)
    ca = "a" if "a" in cols else None
    cz = "z" if "z" in cols else None
    cE2 = "E2" if "E2" in cols else None
    cH  = "H"  if "H"  in cols else None
    if not (ca or cz):
        raise ValueError("table must have 'a' or 'z'")
    if not (cE2 or cH):
        raise ValueError("table must include 'E2' or 'H'")

    A, V = [], []
    for r in rows:
        a = float(r[ca]) if ca else 1.0/(1.0+float(r[cz]))
        v = float(r[cE2]) if cE2 else float(r[cH])
        A.append(a); V.append(v)
    A = np.asarray(A); V = np.asarray(V)
    ordx = np.argsort(A); A, V = A[ordx], V[ordx]
    if cH and not cE2:
        idx = np.argmin(np.abs(A-1.0)); H1 = V[idx]
        V = (V/H1)**2
    idx = np.argmin(np.abs(A-1.0))
    V = V/V[idx]
    return lambda aq: np.interp(aq, A, V)

def H_of_z(z, H0, E2f):
    a = 1.0/(1.0+z)
    return H0 * np.sqrt(max(E2f(a), 1e-300))

def DM_of_z(z, H0, E2f):
    zs = np.linspace(0, z, max(64, int(4096*z)+32))
    Hz = np.array([H_of_z(zi, H0, E2f) for zi in zs])
    integ = np.trapz(1.0/Hz, zs)
    return C_KMS * integ

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lcdm","table"], required=True)
    ap.add_argument("--omega_m0", type=float, help="Ωm0 for lcdm mode")
    ap.add_argument("--H0", type=float, required=True)
    ap.add_argument("--rdrag", type=float, required=True)
    ap.add_argument("--table", type=str, default=None)
    ap.add_argument("--zs", type=str, default="")
    ap.add_argument("--alphas", type=str, default=None)
    ap.add_argument("--out", type=str, default="data/bao_fid_DM_DH.csv")
    args = ap.parse_args()

    if args.mode == "lcdm":
        if args.omega_m0 is None:
            sys.exit("ERROR: --omega_m0 required for lcdm mode")
        E2f = E2_LCDM(args.omega_m0)
    else:
        if not args.table:
            sys.exit("ERROR: --table required for table mode")
        E2f = build_E2_from_table(args.table)

    z_list = []
    if args.zs.strip():
        z_list = [float(z) for z in args.zs.split(",")]
    if args.alphas:
        rows, cols = read_table(args.alphas)
        if "z" not in cols:
            sys.exit("ERROR: --alphas file must contain column 'z'")
        z_list += [float(r["z"]) for r in rows]

    rows_out = []
    for z in sorted(set(z_list)):
        dm = DM_of_z(z, args.H0, E2f) / args.rdrag
        dh = (C_KMS/H_of_z(z, args.H0, E2f)) / args.rdrag
        rows_out.append([z, dm, dh])

    write_fid(args.out, rows_out)

if __name__ == "__main__":
    main()
