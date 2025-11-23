import os, re
import numpy as np
import pandas as pd

def _pick_named(df, names):
    m = {c.lower(): c for c in df.columns}
    for n in names:
        c = m.get(n.lower())
        if c is not None:
            return c
    return None

def _is_number(tok):
    return re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", str(tok)) is not None

def _fallback_numeric_table(path):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"): 
                continue
            toks = [t for t in s.split() if _is_number(t)]
            if len(toks) >= 2:
                rows.append([float(x) for x in toks])
    if not rows:
        raise ValueError("No numeric rows found")
    arr = np.array(rows, float)
    return pd.DataFrame(arr, columns=[f"col{i}" for i in range(arr.shape[1])])

def _read_dat(dat_path, debug=False):
    # Try with header/whitespace; fallback to numeric-only
    try:
        df = pd.read_csv(dat_path, sep=r"\s+", engine="python", comment="#")
        if df.shape[1] < 2:
            raise ValueError("Too few columns")
    except Exception:
        df = _fallback_numeric_table(dat_path)

    # Prefer standard Pantheon+ names, then fallback
    zcol  = _pick_named(df, ["zCMB","zHD","zHEL","z"])
    mucol = _pick_named(df, ["MU_SH0ES","MU","mu","MU_Z","distance_modulus"])
    if zcol is None or mucol is None:
        # heuristic fallback
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) < 2:
            raise ValueError("Could not identify numeric columns for z/mu")
        # Pick z as the numeric column with values mostly in (0,3)
        zcand, best = None, -1e9
        for c in num_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            if v.size==0: 
                continue
            frac = np.mean((v>0) & (v<3))
            med  = np.nanmedian(v)
            score = frac - 0.02*abs(med-0.3)
            if score > best:
                best, zcand = score, c
        zcol = zcol or zcand
        # Pick mu as values ~28–48 mag
        mcand, best = None, -1e9
        for c in num_cols:
            if c == zcol: 
                continue
            v = pd.to_numeric(df[c], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            if v.size==0: 
                continue
            frac = np.mean((v>28) & (v<48))
            med  = np.nanmedian(v)
            std  = np.nanstd(v)
            score = frac - 0.02*abs(med-35) - 0.01*std
            if score > best:
                best, mcand = score, c
        mucol = mucol or mcand

    if debug:
        print(f"[pantheon+] columns: {list(df.columns)}")
        print(f"[pantheon+] picked z='{zcol}', mu='{mucol}' from {os.path.basename(dat_path)}")
        print(df.head(3).to_string(index=False))

    z  = pd.to_numeric(df[zcol], errors='coerce').to_numpy()
    mu = pd.to_numeric(df[mucol], errors='coerce').to_numpy()
    m = np.isfinite(z) & np.isfinite(mu)
    return z[m], mu[m]

def _load_cov(cov_path, N):
    A = np.loadtxt(cov_path)
    # Cases:
    #  (i) NxN matrix
    if A.ndim == 2 and A.shape == (N, N):
        return A
    #  (ii) flattened vector length N^2
    if A.ndim == 1 and A.size == N*N:
        return A.reshape(N, N)
    #  (iii) first entry is N, followed by N^2 values  -> length N^2 + 1
    if A.ndim == 1 and A.size == N*N + 1 and int(round(A[0])) == N:
        return A[1:].reshape(N, N)
    #  (iv) a weird 2D shape but total size N^2
    if A.size == N*N:
        return A.reshape(N, N)
    raise ValueError(f"Covariance shape {A.shape} not compatible with N={N}")

def load_pantheon_plus(dat_path, cov_path, debug=False):
    z, mu = _read_dat(dat_path, debug=debug)
    C = _load_cov(cov_path, mu.size)
    return z, mu, C

def try_all_and_match(dat_files, cov_file, debug=False):
    last_err = None
    for d in dat_files:
        try:
            z, mu = _read_dat(d, debug=debug)
            C = _load_cov(cov_file, mu.size)
            return z, mu, C, d
        except Exception as e:
            last_err = e
            if debug:
                print(f"[pantheon+] {os.path.basename(d)} failed: {e}")
            continue
    raise RuntimeError(f"No Pantheon+ .dat matched covariance using {os.path.basename(cov_file)}. Last error: {last_err}")
