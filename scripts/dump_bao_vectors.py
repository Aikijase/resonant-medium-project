#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, glob, os, re
import numpy as np, pandas as pd
from pathlib import Path

# Import the pure helpers from your runner
from run_snbao_resonant import load_BAO_any, make_grid, distances_LCDM, distances_RES

def bao_model_and_z(kinds, zbao, DM, DH, DV=None):
    """Return model vector m and matching z-values in the same order as `kinds`."""
    m, z = [], []
    iDM = iDH = iDV = 0
    for k in kinds:
        if k == 'DM':
            m.append(DM[iDM]); z.append(zbao[iDM]); iDM += 1
        elif k == 'DH':
            m.append(DH[iDH]); z.append(zbao[iDH]); iDH += 1
        elif k == 'DV':
            if DV is None:
                raise ValueError("DV requested but DV=None")
            m.append(DV[iDV]); z.append(zbao[iDV]); iDV += 1
        else:
            raise ValueError(f"unknown kind {k}")
    return np.asarray(m, float), np.asarray(z, float)

def profile_beta(y, C, m):
    """Profile multiplicative beta with a stable Cholesky inverse."""
    from scipy.linalg import cho_factor, cho_solve
    y = np.asarray(y, float); C = np.asarray(C, float); m = np.asarray(m, float)
    eps = 1e-12 * float(np.median(np.diag(C)))
    Cf  = C + eps * np.eye(C.shape[0])
    cf  = cho_factor(Cf, lower=True, check_finite=False)
    Ci  = lambda v: cho_solve(cf, v, check_finite=False)
    num = float(m @ Ci(y))
    den = float(m @ Ci(m))
    beta = num / den
    r    = y - beta*m
    std  = np.sqrt(np.diag(Cf))
    return beta, r, std

def parse_last_from_logs(tag):
    """Grab latest best-fit params printed by run_snbao_resonant.py logs."""
    logs = sorted(glob.glob("outputs/runs/*.log"), key=os.path.getmtime, reverse=True)
    if tag == "LCDM":
        patt = r"LCDM\s*:\s*\{[^}]*'Om':\s*([-\d.eE+]+)"
    else:
        patt = (r"RESN\s*:\s*\{[^}]*'Om':\s*([-\d.eE+]+)[^}]*'A':\s*([-\d.eE+]+)"
                r"[^}]*'f':\s*([-\d.eE+]+)[^}]*'phi':\s*([-\d.eE+]+)")
    for L in logs:
        try:
            txt = Path(L).read_text(errors="ignore")
        except Exception:
            continue
        m = re.search(patt, txt)
        if m:
            return [float(x) for x in m.groups()]
    return None

def main():
    ap = argparse.ArgumentParser(description="Dump BAO data vs model with profiled β.")
    ap.add_argument("--tag", choices=["LCDM","RESN"], required=True)
    ap.add_argument("--Om", type=float)
    ap.add_argument("--A",  type=float)
    ap.add_argument("--f",  type=float)
    ap.add_argument("--phi",type=float)
    ap.add_argument("--from-logs", dest="from_logs", action="store_true",
                    help="Read latest best-fit params from outputs/runs/*.log")
    args = ap.parse_args()

    # Load BAO data exactly like the runner
    zbao, ybao, kinds, Cbao = load_BAO_any()
    zg, dz = make_grid(zbao, N=2000)

    # Get parameters
    if args.from_logs:
        got = parse_last_from_logs(args.tag)
        if got is None:
            raise SystemExit("Could not parse parameters from logs; pass them explicitly.")
        if args.tag == "LCDM":
            args.Om, = got
        else:
            args.Om, args.A, args.f, args.phi = got

    if args.tag == "LCDM":
        if args.Om is None:
            raise SystemExit("Need --Om (or --from-logs) for LCDM.")
        DM, DH, DV = distances_LCDM(zbao, args.Om, zg, dz)
    else:
        for k in ("Om","A","f","phi"):
            if getattr(args, k) is None:
                raise SystemExit("Need --Om --A --f --phi (or --from-logs) for RESN.")
        DM, DH, DV = distances_RES(zbao, args.Om, args.A, args.f, args.phi, zg, dz)

    m, zvec   = bao_model_and_z(kinds, zbao, DM, DH, DV)
    beta, r, std = profile_beta(ybao, Cbao, m)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "kind": np.array(kinds, dtype=str),
        "z":    zvec,
        "y_data":  np.asarray(ybao, float),
        "y_model": beta*m,
        "residual": r,
        "std_resid": r/std
    }).to_csv(Path(f"outputs/bao_fit_{args.tag}.csv"), index=False)
    print(f"[ok] wrote outputs/bao_fit_{args.tag}.csv  (β={beta:.6f})")

if __name__ == "__main__":
    main()
