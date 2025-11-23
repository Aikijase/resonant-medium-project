#!/usr/bin/env python3
import os, re, sys, glob, argparse
import numpy as np, pandas as pd

DEF_OUTDIR = "data/desi_dr1_bao"
DESI_Z = np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491], float)

def find_candidates():
    roots = [
        "data/desi_dr1_bao",
        "/mnt/chromeos/MyFiles/Downloads",
        "."
    ]
    pats = ["*ALL*GCcomb*mean*", "*ALL*GCcomb*cov*", "*bao*mean*", "*bao*cov*",
            "*BAO*mean*", "*BAO*cov*", "*.csv", "*.txt"]
    found = []
    for r in roots:
        if not os.path.isdir(r): continue
        for p in pats:
            for f in glob.glob(os.path.join(r, p)):
                if os.path.isfile(f):
                    found.append(os.path.abspath(f))
    return sorted(set(found))

def is_cov_name(path): 
    n = os.path.basename(path).lower()
    return ("cov" in n) or ("inverse" in n and "cov" in n)

def load_cov_any(path):
    # Try CSV with headers
    try:
        df = pd.read_csv(path)
        A = df.apply(pd.to_numeric, errors="coerce")
        A = A.loc[:, A.notna().any(0)]
        A = A.loc[A.notna().any(1)]
        C = A.to_numpy(float)
        if C.ndim == 2 and C.shape[0] == C.shape[1]:
            return C
    except Exception:
        pass
    # Try numeric CSV or whitespace
    for delim in [",", None]:
        try:
            C = np.loadtxt(path, delimiter=delim)
            if C.ndim == 2 and C.shape[0] == C.shape[1]:
                return C
        except Exception:
            continue
    raise RuntimeError(f"Could not parse covariance from {path}")

def read_mean_vector(path):
    y = np.loadtxt(path, dtype=float).ravel()
    if y.size not in (12,18): 
        raise ValueError("not 12/18-length vector")
    return y

def read_table(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    cols = {c.lower(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            if o in cols: return cols[o]
        return None
    kz  = pick("z","zeff","z_eff")
    kDM = pick("dm_over_r_d","dm_over_rs","d_m_over_r_d","dm/rd","dm/rs")
    kDH = pick("dh_over_r_d","dh_over_rs","d_h_over_r_d","dh/rd","dh/rs")
    kDV = pick("dv_over_r_d","dv_over_rs","d_v_over_r_d","dv/rd","dv/rs")
    if kz is None or kDM is None or kDH is None:
        raise ValueError("missing z/DM/DH columns")
    out = {"mode":"table",
           "z": df[kz].to_numpy(float),
           "DM": df[kDM].to_numpy(float),
           "DH": df[kDH].to_numpy(float)}
    if kDV is not None:
        out["DV"] = df[kDV].to_numpy(float)
    return out

def choose_dm_dh_from_vector(y):
    nb = 6
    haveDV = (y.size == 18)
    # Packing A: [DM(z1..z6), DH(z1..z6), (DV...)]
    DM_A, DH_A = y[:nb], y[nb:2*nb]
    # Packing B: [DM1, DH1, (DV1), DM2, DH2, (DV2)...]
    step = 3 if haveDV else 2
    DM_B, DH_B = y[0::step][:nb], y[1::step][:nb]
    mono = lambda a: (np.diff(a) > 0).sum()
    return (DM_A, DH_A) if mono(DM_A) >= mono(DM_B) else (DM_B, DH_B), haveDV

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean", help="path to DESI BAO mean file (vector or table)")
    ap.add_argument("--cov",  help="path to DESI BAO covariance file")
    ap.add_argument("--outdir", default=DEF_OUTDIR)
    ap.add_argument("--auto", action="store_true", help="search common locations")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    mean_path, cov_path = args.mean, args.cov
    if args.auto or not (mean_path and cov_path):
        cands = find_candidates()
        # Pick first plausible mean & cov
        for f in cands:
            if not mean_path:
                try:
                    _ = read_mean_vector(f); mean_path = f
                except Exception:
                    try:
                        _ = read_table(f); mean_path = f
                    except Exception:
                        pass
            if not cov_path and is_cov_name(f):
                try:
                    _ = load_cov_any(f); cov_path = f
                except Exception:
                    pass
            if mean_path and cov_path:
                break

    if not mean_path or not cov_path:
        print("[ERR] Could not find mean/cov. Supply --mean and --cov explicitly.")
        print("      Searched:", find_candidates())
        sys.exit(1)

    # Parse mean
    try:
        y = read_mean_vector(mean_path)
        (DM, DH), haveDV = choose_dm_dh_from_vector(y)
        # slice covariance to DM+DH only
        Cfull = load_cov_any(cov_path)
        nb = 6; step = 3 if haveDV else 2
        # detect which packing we chose
        if np.allclose(DM, y[:nb]):
            idx_DM = list(range(0, nb))
            idx_DH = list(range(nb, 2*nb))
        else:
            idx_DM = list(range(0, step*nb, step))
            idx_DH = list(range(1, step*nb, step))
        idx = np.array(idx_DM + idx_DH, int)
        C = Cfull[np.ix_(idx, idx)]
        z = DESI_Z
    except Exception:
        # Try as table with headers
        t = read_table(mean_path)
        z, DM, DH = t["z"], t["DM"], t["DH"]
        Cfull = load_cov_any(cov_path)
        n = Cfull.shape[0]; nb = len(z)
        if n == 2*nb:
            idx = np.r_[np.arange(0, nb), np.arange(nb, 2*nb)]
        elif n == 3*nb:
            # assume block [DM, DH, DV]
            idx = np.r_[np.arange(0, nb), np.arange(nb, 2*nb)]
        else:
            # interleaved fallback
            step = 3 if n == 3*nb else 2
            idx_DM = list(range(0, step*nb, step))
            idx_DH = list(range(1, step*nb, step))
            idx = np.array(idx_DM + idx_DH, int)
        C = Cfull[np.ix_(idx, idx)]

    # Save outputs
    df = pd.DataFrame({"z": z, "DM_over_r_d": DM, "DH_over_r_d": DH})
    out_meas = os.path.join(args.outdir, "bao_measurements.csv")
    out_cov  = os.path.join(args.outdir, "bao_covariance.csv")
    df.to_csv(out_meas, index=False)
    np.savetxt(out_cov, C, delimiter=",")

    print(f"[OK] wrote {out_meas} (rows={len(df)}) and {out_cov} (shape={C.shape})")
    print(f"[INFO] mean source: {mean_path}")
    print(f"[INFO] cov  source: {cov_path}")

if __name__ == "__main__":
    main()
