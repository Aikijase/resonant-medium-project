import argparse, numpy as np, pandas as pd, sys, pathlib as p
from numpy.linalg import eigh, cholesky
ap = argparse.ArgumentParser()
ap.add_argument("--in", required=True, dest="cin")
ap.add_argument("--sn-csv", required=True)
ap.add_argument("--out", required=True, dest="cout")
ap.add_argument("--floor-frac", type=float, default=1e-5)
args = ap.parse_args()
cin, cout = p.Path(args.cin).expanduser(), p.Path(args.cout).expanduser()
N = len(pd.read_csv(p.Path(args.sn_csv).expanduser()))
try:
    C = np.loadtxt(cin, delimiter=",")
except Exception:
    C = np.loadtxt(cin)  # whitespace
if C.ndim == 1:
    m = C.size
    if m == N*N:
        C = C.reshape(N, N)
    elif m == N*N + 1:
        C = C[:N*N].reshape(N, N)
    else:
        sys.exit(f"[error] 1D cov length {m} != N^2 ({N*N})")
if C.shape != (N,N): sys.exit(f"[error] cov shape {C.shape} != ({N},{N})")
C = (C + C.T)/2.0
w, V = eigh(C)
w_med = float(np.median(w))
floor = max(args.floor_frac * w_med, 1e-12)
w2 = np.maximum(w, floor)
C2 = (V * w2) @ V.T
C2 = (C2 + C2.T)/2.0
# verify SPD
try:
    cholesky(C2)
except Exception as e:
    sys.exit(f"[error] post-floor Cholesky failed: {e}")
np.savetxt(cout, C2, fmt="%.10e", delimiter=",")
print(f"[ok] wrote {cout} shape: {C2.shape}  floor={floor:.3e}")
