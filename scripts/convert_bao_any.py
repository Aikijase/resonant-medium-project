#!/usr/bin/env python3
import os, glob, re, argparse
import numpy as np, pandas as pd

OUTDIR = "data/desi_dr1_bao"
DESI_Z = np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491], float)

def read_cov_any(path):
    # Try CSV with headers -> coerce numerics
    try:
        df = pd.read_csv(path)
        A = df.apply(pd.to_numeric, errors="coerce")
        A = A.loc[:, A.notna().any(0)]
        A = A.loc[A.notna().any(1)]
        C = A.to_numpy(float)
        if C.ndim==2 and C.shape[0]==C.shape[1]:
            return C
    except Exception:
        pass
    # Try numeric CSV or whitespace
    for delim in [",", None]:
        try:
            C = np.loadtxt(path, delimiter=delim)
            if C.ndim==2 and C.shape[0]==C.shape[1]:
                return C
        except Exception:
            continue
    raise RuntimeError(f"Could not parse covariance from {path}")

def canon_qty(s):
    s = re.sub(r'[^A-Za-z]', '', str(s)).lower()
    if s.startswith('dm'): return 'DM'
    if s.startswith('dh'): return 'DH'
    if s.startswith('dv'): return 'DV'
    return None

def read_mean_any(path):
    # (A) try numeric vector (length 12 or 18)
    try:
        y = np.loadtxt(path, dtype=float).ravel()
        if y.size in (12,18):
            return {"mode":"vector", "y":y}
    except Exception:
        pass

    # (B) try wide table with headers
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
        except Exception:
            df = None
    if df is not None and df.shape[1] >= 2:
        cols = {c.lower(): c for c in df.columns}
        def pick(*opts):
            for o in opts:
                if o in cols: return cols[o]
            return None
        kz  = pick("z","zeff","z_eff")
        kDM = pick("dm_over_r_d","dm_over_rs","d_m_over_r_d","dm/rd","dm/rs")
        kDH = pick("dh_over_r_d","dh_over_rs","d_h_over_r_d","dh/rd","dh/rs")
        kDV = pick("dv_over_r_d","dv_over_rs","d_v_over_r_d","dv/rd","dv/rs")
        if kz and kDM and kDH:
            out = {"mode":"wide",
                   "z":  df[kz].to_numpy(float),
                   "DM": df[kDM].to_numpy(float),
                   "DH": df[kDH].to_numpy(float)}
            if kDV: out["DV"] = df[kDV].to_numpy(float)
            return out

    # (C) LONG format: 3 cols: z, value, quantity (possibly with '#' comments)
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=["z","value","quantity"], comment="#",
                         dtype={"z":float,"value":float,"quantity":str})
        # Keep only rows with a recognizable quantity
        df["kind"] = df["quantity"].map(canon_qty)
        df = df[df["kind"].isin(["DM","DH","DV"])].copy()
        if df.empty: raise RuntimeError("no DM/DH/DV rows")
        # Preserve file order; we’ll also sort by z when we construct DM/DH arrays
        return {"mode":"long", "df": df}
    except Exception:
        pass

    raise RuntimeError(f"Unrecognized mean format for {path}")

def choose_dm_dh_from_vector(y, Cfull):
    nb = 6
    haveDV = (y.size == 18)
    # Packing A: [DM(z1..z6), DH(z1..z6), (DV...)]
    DM_A, DH_A = y[:nb], y[nb:2*nb]
    # Packing B: interleaved [DM1, DH1, (DV1), DM2, DH2, (DV2), ...]
    step = 3 if haveDV else 2
    DM_B, DH_B = y[0::step][:nb], y[1::step][:nb]
    mono = lambda a: (np.diff(a) > 0).sum()
    DM, DH = (DM_A, DH_A) if mono(DM_A) >= mono(DM_B) else (DM_B, DH_B)

    # Covariance slice to DM+DH only, matching chosen packing
    if y.size == 12:
        idx = np.r_[np.arange(0,6), np.arange(6,12)]
    else:
        idx = np.array(list(range(0,18,3)) + list(range(1,18,3)))
    Cred = Cfull[np.ix_(idx, idx)]
    return DESI_Z, DM, DH, Cred

def build_from_wide(z, DM, DH, Cfull):
    nb = len(z); n = Cfull.shape[0]
    # Covariance could be [DM(z..), DH(z..)] blocks (2*nb) or include DV (3*nb)
    if n == 2*nb:
        idx = np.r_[np.arange(0,nb), np.arange(nb,2*nb)]
    elif n == 3*nb:
        idx = np.r_[np.arange(0,nb), np.arange(nb,2*nb)]  # drop DV block
    else:
        # Interleaved fallback
        step = 3 if n == 3*nb else 2
        idx = np.array(list(range(0,step*nb,step)) + list(range(1,step*nb,step)))
    Cred = Cfull[np.ix_(idx, idx)]
    # Sort by z for output consistency
    order = np.argsort(z)
    return z[order], DM[order], DH[order], Cred[np.ix_(np.r_[order, order+len(order)],
                                                      np.r_[order, order+len(order)])]

def build_from_long(df, Cfull):
    """
    df columns: z, value, quantity, kind in file order.
    We keep original order for mapping to covariance rows, then
    reorder to [DM(z-ascending), DH(z-ascending)] for output and
    permute covariance accordingly.
    """
    # Positions in file-order
    idx_DM_full = df.index[df["kind"]=="DM"].to_numpy()
    idx_DH_full = df.index[df["kind"]=="DH"].to_numpy()
    # Values and z in file-order
    z_dm = df.loc[idx_DM_full, "z"].to_numpy(float)
    v_dm = df.loc[idx_DM_full, "value"].to_numpy(float)
    z_dh = df.loc[idx_DH_full, "z"].to_numpy(float)
    v_dh = df.loc[idx_DH_full, "value"].to_numpy(float)

    # Sanity: need same # of DM and DH bins
    nb = min(len(z_dm), len(z_dh))
    z_dm, v_dm = z_dm[:nb], v_dm[:nb]
    z_dh, v_dh = z_dh[:nb], v_dh[:nb]
    # Sort z bins ascending for output
    ord_dm = np.argsort(z_dm); ord_dh = np.argsort(z_dh)
    z_out  = z_dm[ord_dm]  # assume DM and DH share the same zeff set
    DM_out = v_dm[ord_dm]; DH_out = v_dh[ord_dh]

    # Build covariance permutation: first DM(z-asc), then DH(z-asc)
    idx_dm_sorted = idx_DM_full[ord_dm]
    idx_dh_sorted = idx_DH_full[ord_dh]
    select = np.r_[idx_dm_sorted, idx_dh_sorted]
    Cred = Cfull[np.ix_(select, select)]
    return z_out, DM_out, DH_out, Cred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean", required=True, help="path to DESI BAO mean (vector, wide, or long format)")
    ap.add_argument("--cov",  required=True, help="path to DESI BAO covariance")
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    m = read_mean_any(args.mean)
    Cfull = read_cov_any(args.cov)

    if m["mode"] == "vector":
        z, DM, DH, Cred = choose_dm_dh_from_vector(m["y"], Cfull)
    elif m["mode"] == "wide":
        z, DM, DH, Cred = build_from_wide(m["z"], m["DM"], m["DH"], Cfull)
    else:  # long
        z, DM, DH, Cred = build_from_long(m["df"], Cfull)

    # Write outputs expected by the runner
    out_meas = os.path.join(args.outdir, "bao_measurements.csv")
    out_cov  = os.path.join(args.outdir, "bao_covariance.csv")
    pd.DataFrame({"z": z, "DM_over_r_d": DM, "DH_over_r_d": DH}).to_csv(out_meas, index=False)
    np.savetxt(out_cov, Cred, delimiter=",")

    print(f"[OK] wrote {out_meas} (rows={len(z)}) and {out_cov} (shape={Cred.shape})")

if __name__ == "__main__":
    main()
