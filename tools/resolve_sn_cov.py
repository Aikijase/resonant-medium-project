import sys, os, numpy as np, pandas as pd

if len(sys.argv) < 4:
    print("usage: python tools/resolve_sn_cov.py <sn_vec_csv> <sn_cov_in> <sn_cov_out>")
    sys.exit(2)

SN_VEC, SN_COV_IN, SN_COV_OUT = sys.argv[1:4]
os.makedirs(os.path.dirname(SN_COV_OUT) or ".", exist_ok=True)

# 1) read SN vector to get N
sn = pd.read_csv(SN_VEC)
N = sn.shape[0]

# 2) try dataframe first (often works for CSV squares)
try:
    df = pd.read_csv(SN_COV_IN, header=None)
    arr = df.values
    if arr.ndim == 2 and arr.shape == (N, N):
        C = arr.astype(float)
        kind = "square(csv)"
    else:
        raise Exception("not square via pandas")
except Exception:
    # fallback: raw numeric load
    vec = np.loadtxt(SN_COV_IN)
    v = np.squeeze(vec)
    if v.ndim == 2 and v.shape == (N, N):
        C = v.astype(float); kind = "square(txt)"
    elif v.ndim == 1:
        M = v.size
        if M == N:
            C = np.diag(v); kind = "diag->square"
        elif M == N*N:
            C = v.reshape(N, N); kind = "flat N^2 -> square"
        elif M == N*N + 1:
            C = v[:-1].reshape(N, N); kind = "flat N^2+1 -> dropped last -> square"
        elif M == N*(N+1)//2:
            # unpack packed upper triangle
            C = np.zeros((N,N), float); k = 0
            for i in range(N):
                for j in range(i, N):
                    C[i,j] = C[j,i] = v[k]; k += 1
            kind = "packed SPD -> square"
        else:
            raise ValueError(f"1-D cov length {M} not in {{N={N}, N^2={N*N}, N^2+1, N(N+1)/2={N*(N+1)//2}}}")
    else:
        raise ValueError(f"Unsupported cov array ndim={v.ndim}")

# tiny jitter if needed
wmin = np.linalg.eigvalsh(C).min()
if wmin <= 0:
    eps = abs(wmin) + 1e-12
    C = C + np.eye(N)*eps
    print(f"[resolve_sn_cov] added jitter {eps:.3e}; new min eig={np.linalg.eigvalsh(C).min():.3e}")
np.savetxt(SN_COV_OUT, C, fmt="%.8e", delimiter=",")
print(f"[resolve_sn_cov] N={N}  wrote {SN_COV_OUT}  shape={C.shape}  source={kind}")
