import argparse, numpy as np, pandas as pd, pathlib as p, sys
from numpy.linalg import cholesky, eigh

ap = argparse.ArgumentParser()
ap.add_argument("--cov-in", required=True)
ap.add_argument("--sn-csv", required=True)           # expects columns: z,mu,sigma
ap.add_argument("--cov-out", required=True)
ap.add_argument("--floor-frac", type=float, default=1e-6,
                help="eigenvalue floor as fraction of median eigenvalue (after scaling)")
args = ap.parse_args()

cov_in  = p.Path(args.cov_in).expanduser()
cov_out = p.Path(args.cov_out).expanduser()
sn_csv  = p.Path(args.sn_csv).expanduser()

df = pd.read_csv(sn_csv)
N = len(df)
sig2_target = float(np.median(np.asarray(df["sigma"], float)**2))

# Load and reshape covariance
try:
    C = np.loadtxt(cov_in, delimiter=",")
except Exception:
    C = np.loadtxt(cov_in)
if C.ndim == 1:
    m = C.size
    if   m == N*N:       C = C.reshape(N,N)
    elif m == N*N + 1:   C = C[:N*N].reshape(N,N)
    else:
        sys.exit(f"[error] 1D cov length {m} != N^2 ({N*N})")
elif C.shape != (N,N):
    sys.exit(f"[error] cov shape {C.shape} != ({N},{N})")

# Symmetrize
C = (C + C.T)/2.0

# Scale so median diag matches target sigma^2
d = np.diag(C)
if not np.all(d > 0):
    sys.exit("[error] covariance has non-positive diagonal before scaling.")
scale = sig2_target / float(np.median(d))
C = C * scale

# Floor tiny eigenvalues to ensure SPD
w, V = eigh((C + C.T)/2.0)
w_med = float(np.median(w))
floor = max(args.floor_frac * w_med, 1e-12)
w2 = np.maximum(w, floor)
C2 = (V * w2) @ V.T
C2 = (C2 + C2.T)/2.0

# Final SPD check
cholesky(C2)

np.savetxt(cov_out, C2, fmt="%.10e", delimiter=",")
print(f"[ok] wrote {cov_out}  shape={C2.shape}  scale={scale:.3e}  floor={floor:.3e}")
