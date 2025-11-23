#!/usr/bin/env python3
"""
bg_apply_bumps.py
Apply one or more smooth Gaussian bumps to E2(a)=H(a)^2/H0^2 in a background CSV (a,E2 or a,H).

Example:
  python tools/bg_apply_bumps.py \
    --in data/bg.csv --out data/bg.tuned2.csv \
    --bump a0=0.432,width=0.08,H_mul=1.19 \
    --bump a0=0.518,width=0.08,H_mul=1.08
"""
import csv, argparse, math
from pathlib import Path
import numpy as np

def read_rows(p):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r); cols = r.fieldnames
    return rows, cols

def parse_bump(spec):
    # spec like a0=0.432,width=0.08,H_mul=1.19
    kv = dict(x.split("=") for x in spec.split(","))
    return float(kv["a0"]), float(kv.get("width", 0.08)), float(kv["H_mul"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--bump", action="append", required=True,
                    help="a0=<center a>,width=<sigma in a>,H_mul=<H multiplier at center>")
    args = ap.parse_args()

    rows, cols = read_rows(args.inp)
    if "a" not in cols: raise SystemExit("background must include column 'a'")
    has_E2 = "E2" in cols; has_H = "H" in cols
    if not (has_E2 or has_H): raise SystemExit("background must include 'E2' or 'H'")

    A = np.array([float(r["a"]) for r in rows])
    if has_E2:
        E2 = np.array([float(r["E2"]) for r in rows])
    else:
        H  = np.array([float(r["H"]) for r in rows])
        H1 = H[np.argmin(np.abs(A-1.0))]
        E2 = (H/H1)**2

    E2_new = E2.copy()
    for spec in args.bump:
        a0, width, H_mul = parse_bump(spec)
        bump = np.exp(-0.5*((A-a0)/width)**2)             # Gaussian in a
        E2_new *= (1.0 + (H_mul**2 - 1.0)*bump)           # because E2 ∝ H^2

    # Re-normalize to E2(a=1)=1
    E2_new /= E2_new[np.argmin(np.abs(A-1.0))]

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a","E2"])
        for a,e2 in zip(A, E2_new):
            w.writerow([f"{a:.9f}", f"{e2:.9f}"])
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
