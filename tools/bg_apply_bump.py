#!/usr/bin/env python3
"""
Apply a smooth local bump to E2(a) in a background CSV (a,E2 or a,H),
writing a new table for quick BAO what-if tests.

Usage:
  python tools/bg_apply_bump.py \
    --in data/bg.csv --out data/bg.tuned.csv \
    --a0 0.432 --width 0.08 --H_mul 1.19

Notes:
  - If the CSV has H instead of E2, we convert to E2, bump it, then output E2.
  - Bump shape: exp(-0.5*((a-a0)/width)^2) applied to H by factor H_mul at peak.
  - Because E2 = H^2/H0^2, we actually multiply E2 by (H_mul^2) at the peak.
"""
import csv, argparse, math
from pathlib import Path
import numpy as np

def read_rows(p):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r); cols = r.fieldnames
    return rows, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--a0", type=float, required=True, help="center in a")
    ap.add_argument("--width", type=float, default=0.08, help="Gaussian width in a")
    ap.add_argument("--H_mul", type=float, required=True, help="H multiplier at a0 (e.g., 1.19)")
    args = ap.parse_args()

    rows, cols = read_rows(args.inp)
    ca = "a" if "a" in cols else None
    cE2 = "E2" if "E2" in cols else None
    cH  = "H"  if "H"  in cols else None
    if not ca: raise SystemExit("background must include column 'a'")
    if not (cE2 or cH): raise SystemExit("background must include 'E2' or 'H'")

    A = np.array([float(r[ca]) for r in rows])
    if cE2:
        E2 = np.array([float(r[cE2]) for r in rows])
    else:
        H  = np.array([float(r[cH]) for r in rows])
        # normalise H by H(a=1) to get E
        idx = np.argmin(np.abs(A-1.0)); H1 = H[idx]
        E2 = (H/H1)**2

    # Build Gaussian bump on H -> multiply E2 by (H_mul^2) at center
    Hmul = args.H_mul
    bump = np.exp(-0.5*((A-args.a0)/args.width)**2)
    E2_tuned = E2 * (1.0 + (Hmul**2 - 1.0)*bump)

    # Renormalize so E2(a=1)=1 (just in case)
    idx1 = np.argmin(np.abs(A-1.0))
    E2_tuned /= E2_tuned[idx1]

    # Write out (a,E2) table
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a","E2"])
        for a, e2 in zip(A, E2_tuned):
            w.writerow([f"{a:.9f}", f"{e2:.9f}"])
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
