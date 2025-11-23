#!/usr/bin/env python3
"""
Find a plausible residual series in outputs/phase8 (or nearby) and export to:
  - outputs/phase8/residuals.csv  (columns: x,res)
  - outputs/phase8/residual_series.npz  (arrays: x, res)

Heuristics:
- Accepts CSV/JSON/NPZ with keys/columns similar to:
    x, efold, k   and   res, residual, R, Pk_res
- If only a single residual-like vector is found, x becomes arange(n).
- Picks the longest usable (x,res) pair across candidates.
"""
import os, json, pathlib, numpy as np, csv, sys, glob

CAND_DIRS = [
    "outputs/phase8",
    "outputs",
    "outputs/phase8_spectral",
]
CAND_PATTERNS = [
    "*.npz", "*.json", "*.csv",
]

def try_load(path):
    p = pathlib.Path(path)
    try:
        if p.suffix.lower()==".npz":
            Z = np.load(p)
            # Try pairs
            for xk, rk in (("x","res"),("efold","res"),("k","Pk_res"),("x","R")):
                if xk in Z.files and rk in Z.files:
                    x = np.asarray(Z[xk]); r = np.asarray(Z[rk])
                    return x, r, f"npz:{xk},{rk}"
            # If there is just one residual-like vector
            for rk in ("res","residual","R","Pk_res"):
                if rk in Z.files:
                    r = np.asarray(Z[rk]); x = np.arange(r.size)
                    return x, r, f"npz:arange,{rk}"
        elif p.suffix.lower()==".json":
            J = json.load(open(p))
            if "x" in J and ("res" in J or "R" in J):
                return np.asarray(J["x"]), np.asarray(J.get("res", J["R"])), "json:x,res"
            if "k" in J and "Pk_res" in J:
                return np.asarray(J["k"]), np.asarray(J["Pk_res"]), "json:k,Pk_res"
            # single vector
            for rk in ("res","residual","R","Pk_res"):
                if rk in J and isinstance(J[rk], list):
                    r = np.asarray(J[rk]); x = np.arange(r.size)
                    return x, r, f"json:arange,{rk}"
        elif p.suffix.lower()==".csv":
            xs, rs = [], []
            with open(p) as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    x = row.get("x") or row.get("efold") or row.get("k")
                    r = row.get("res") or row.get("residual") or row.get("R") or row.get("Pk_res")
                    if r is None: continue
                    rs.append(float(r))
                    xs.append(float(x) if x is not None else (len(xs)))
            if rs:
                x = np.asarray(xs if all(v is not None for v in xs) else np.arange(len(rs)))
                r = np.asarray(rs)
                return x, r, "csv:x?,res"
    except Exception:
        pass
    return None

def main():
    cands = []
    for d in CAND_DIRS:
        for pat in CAND_PATTERNS:
            cands += glob.glob(os.path.join(d, pat))
    # Try more specific names first
    cands = sorted(cands, key=lambda s: (0 if "phase8" in s else 1, len(s)))
    best = None
    for fn in cands:
        got = try_load(fn)
        if got is None: continue
        x, r, how = got
        if x is None or r is None: continue
        if x.size != r.size: 
            # allow x missing -> arange; otherwise skip
            if x.size==0 or r.size==0: continue
        if best is None or r.size > best[0].size:
            best = (r, x, fn, how)
    if best is None:
        sys.stderr.write("No usable residuals found. Provide a file with x,res or k,Pk_res.\n")
        sys.exit(2)
    r, x, src, how = best
    outdir = pathlib.Path("outputs/phase8"); outdir.mkdir(parents=True, exist_ok=True)
    # CSV
    csvpath = outdir/"residuals.csv"
    with open(csvpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x","res"])
        for xi, ri in zip(x, r): w.writerow([float(xi), float(ri)])
    # NPZ
    npzpath = outdir/"residual_series.npz"
    np.savez(npzpath, x=x.astype(float), res=r.astype(float))
    print("Exported residuals from", src, "via", how)
    print(" ->", csvpath)
    print(" ->", npzpath)

if __name__ == "__main__":
    main()
