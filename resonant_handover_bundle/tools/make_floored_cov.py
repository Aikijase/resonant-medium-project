#!/usr/bin/env python3
import argparse, numpy as np, pathlib as p
from numpy.linalg import eigh, cholesky

def main():
    ap = argparse.ArgumentParser(description="Eigen-floor a covariance to enforce SPD.")
    ap.add_argument("--in", dest="inp", required=True, help="Input STAT+SYS covariance (.cov or .csv)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path for floored SPD covariance")
    ap.add_argument("--floor-frac", type=float, default=1e-6, help="Fraction of median eigenvalue used as floor (default 1e-6)")
    args = ap.parse_args()

    inp = p.Path(args.inp).expanduser()
    out = p.Path(args.out).expanduser()

    # Load (CSV first; fallback to whitespace)
    try:
        C = np.loadtxt(inp, delimiter=",")
    except Exception:
        C = np.loadtxt(inp)

    if C.ndim == 1:
        m = C.size
        N = int(round(np.sqrt(m)))
        if N*N == m:
            C = C.reshape(N, N)
        elif N*N + 1 == m:
            C = C[:-1].reshape(N, N)
            print("[fix] dropped 1 trailing element to square the matrix")
        else:
            raise SystemExit(f"[error] Cannot reshape size {m} into NxN.")

    # Symmetrize
    C = 0.5*(C + C.T)

    # Eigen-floor
    w, V = eigh(C)
    w_med = np.median(w)
    floor = max(args.floor_frac * w_med, 1e-10)
    w2 = np.maximum(w, floor)
    C2 = (V * w2) @ V.T

    # Diagnostics
    cond_orig = (w.max()/w.min()) if np.all(w>0) else np.inf
    cond_new  = (w2.max()/w2.min())
    print(f"[diag] cond(original)~{cond_orig:,.3e}  cond(floored)~{cond_new:,.3e}")

    try:
        cholesky(C2)
        print(f"[ok] Cholesky success with floor = {floor:g}")
    except Exception as e:
        print("[warn] Cholesky still fails after flooring:", e)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, C2, delimiter=",")
    print(f"[ok] wrote {out}  shape: {C2.shape}")

if __name__ == "__main__":
    main()
