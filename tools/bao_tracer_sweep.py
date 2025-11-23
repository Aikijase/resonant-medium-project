#!/usr/bin/env python3
import argparse, re
import numpy as np, pandas as pd
from scipy import integrate

C_KMS = 299792.458

def Ez(z, om0): return np.sqrt(om0*(1+z)**3 + (1-om0))
def DM_Mpc(z, om0, h):
    H0 = 100.0*h
    z = np.atleast_1d(z)
    integ = np.array([integrate.quad(lambda zz: 1.0/Ez(zz, om0), 0.0, zi, limit=256)[0] for zi in z])
    return (C_KMS/H0)*integ
def DH_Mpc(z, om0, h):
    H0 = 100.0*h
    z = np.atleast_1d(z)
    return (C_KMS/H0)/Ez(z, om0)
def rd_Mpc(om0, h, obh2=0.02237):
    omh2 = om0*h*h
    return 55.154*np.exp(-72.3*(obh2+0.0006)**2)/(omh2**0.25351 * (obh2**0.12807))

# -------- column detection --------
def autodetect(df):
    cols = {}
    for k in ['z','Z','redshift']: 
        if k in df.columns: cols['z']=k; break
    for k in ['name','tracer','sample','dataset','survey','label']:
        if k in df.columns: cols['label']=k; break
    # wide-form possibilities
    wide_map = {
        'DM_over_rd': ['DM_over_rd','DM/rd','DMrd','DM_over_rd_divh','DM_over_rd_mulh'],
        'DH_over_rd': ['DH_over_rd','DH/rd','DHrd','DH_over_rd_divh','DH_over_rd_mulh'],
        'DV_over_rd': ['DV_over_rd','DV/rd','DVrd','DV_over_rd_divh','DV_over_rd_mulh'],
    }
    for key, cands in wide_map.items():
        for c in cands:
            if c in df.columns: cols[key]=c; break
    # long-form possibilities
    for k in ['y','value','val','meas','measurement']:
        if k in df.columns: cols['y']=k; break
    for k in ['observable','obs','kind','type','mode','quantity']:
        if k in df.columns: cols['kind']=k; break
    return cols

def read_cov(path):
    if path.lower() in ('none','auto','diag'): return None
    if path.endswith('.csv'): return np.loadtxt(path, delimiter=",")
    return np.loadxt(path)

def maybe_diag_cov(df, cols, cov):
    # Build diagonal cov if mismatch and sigma is present
    if cov is None: pass
    try:
        n = len(df)
        if cov is not None and getattr(cov, 'shape', (0,0)) == (n,n): return cov
        if 'sigma' in df.columns:
            sig = df['sigma'].to_numpy(float)
            return np.diag(sig*sig)
    except Exception:
        pass
    return cov

    if path.endswith('.csv'): return np.loadtxt(path, delimiter=",")
    return np.loadtxt(path)

# -------- model vector given df rows --------
def model_vector(df, cols, om0, h):
    z = df[cols['z']].to_numpy(float)
    rd = rd_Mpc(om0, h)
    # LONG form: per-row kind + y
    if 'y' in cols:
        if 'kind' in cols:
            kinds = df[cols['kind']].astype(str).str.lower().values
            m = np.empty_like(z, dtype=float)
            # compute per-row depending on kind token
            DM = DM_Mpc(z, om0, h)/rd
            DH = DH_Mpc(z, om0, h)/rd
            # simple DV from DM,DH
            DV = ((DM_Mpc(z, om0, h)**2)*DH_Mpc(z, om0, h))**(1/3.0)/rd
            for i, k in enumerate(kinds):
                if   'dm' in k: m[i] = DM[i]
                elif 'dh' in k: m[i] = DH[i]
                elif 'dv' in k: m[i] = DV[i]
                else:           m[i] = DM[i]  # default to DM if unknown
        else:
            # no kind column -> assume DM/rd long vector
            m = DM_Mpc(z, om0, h)/rd
        y = df[cols['y']].to_numpy(float)
        order = None  # not needed in row-wise path
        n = len(df)
        return m, y, order, n
    # WIDE form: concatenate blocks in fixed DM, DH, DV order
    blocks_m, blocks_y, order = [], [], []
    if 'DM_over_rd' in cols:
        DMrd = DM_Mpc(z, om0, h)/rd
        blocks_m.append(DMrd); blocks_y.append(df[cols['DM_over_rd']].to_numpy(float)); order.append('DM')
    if 'DH_over_rd' in cols:
        DHrd = DH_Mpc(z, om0, h)/rd
        blocks_m.append(DHrd); blocks_y.append(df[cols['DH_over_rd']].to_numpy(float)); order.append('DH')
    if 'DV_over_rd' in cols:
        DV = ((DM_Mpc(z, om0, h)**2)*DH_Mpc(z, om0, h))**(1/3.0)/rd
        blocks_m.append(DV); blocks_y.append(df[cols['DV_over_rd']].to_numpy(float)); order.append('DV')
    if not blocks_m:
        raise RuntimeError("Could not detect BAO columns (neither long-form 'y' nor wide-form *_over_rd).")
    return np.concatenate(blocks_m), np.concatenate(blocks_y), order, len(z)

def chi2_with_scales(df, cov, cols, om0, h, hELG, hLRG2, elg_pat, lrg2_pat):
    m, y, order, n = model_vector(df, cols, om0, h)
    cov = maybe_diag_cov(df, cols, cov)
    # row-wise label handling
    labels = df[cols['label']].astype(str).values if 'label' in cols else np.array(['']*n)
    is_elg  = np.array([bool(re.search(elg_pat, s, re.I)) for s in labels])
    is_lrg2 = np.array([bool(re.search(lrg2_pat, s, re.I)) for s in labels])
    m = m.copy()
    if order is None:
        # long-form: masks apply 1:1 to rows
        m[is_elg]  *= hELG
        m[is_lrg2] *= hLRG2
    else:
        # wide-form: replicate mask per block
        rep_elg  = np.concatenate([is_elg  for _ in order])
        rep_lrg2 = np.concatenate([is_lrg2 for _ in order])
        m[rep_elg]  *= hELG
        m[rep_lrg2] *= hLRG2
    if cov.shape != (y.size, y.size):
        raise RuntimeError(f"Covariance shape {cov.shape} != data length {y.size}")
    r = y - m
    try: inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError: inv = np.linalg.pinv(cov)
    chi2 = float(r.T @ inv @ r)
    dof  = y.size - 2
    return chi2, dof, (chi2/dof if dof>0 else np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bao-csv', required=True)
    ap.add_argument('--cov-csv', required=True)
    ap.add_argument('--om0', type=float, required=True)
    ap.add_argument('--h',   type=float, required=True)
    # optional explicit overrides
    ap.add_argument('--z-col'); ap.add_argument('--label-col')
    ap.add_argument('--y-col'); ap.add_argument('--kind-col')
    ap.add_argument('--dm-col'); ap.add_argument('--dh-col'); ap.add_argument('--dv-col')
    # sweep / fixed
    ap.add_argument('--hELG-range',  nargs=3, type=float, metavar=('MIN','MAX','STEPS'))
    ap.add_argument('--hLRG2-range', nargs=3, type=float, metavar=('MIN','MAX','STEPS'))
    ap.add_argument('--hELG-fixed', type=float); ap.add_argument('--hLRG2-fixed', type=float)
    ap.add_argument('--elg-pattern', default='ELG'); ap.add_argument('--lrg2-pattern', default='LRG2')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--debug-cols', action='store_true')
    args = ap.parse_args()

    df  = pd.read_csv(args.bao_csv)
    cov = read_cov(args.cov_csv)
    cols = autodetect(df)
    # apply overrides if provided
    if args.z_col:     cols['z']     = args.z_col
    if args.label_col: cols['label'] = args.label_col
    if args.y_col:     cols['y']     = args.y_col
    if args.kind_col:  cols['kind']  = args.kind_col
    if args.dm_col:    cols['DM_over_rd'] = args.dm_col
    if args.dh_col:    cols['DH_over_rd'] = args.dh_col
    if args.dv_col:    cols['DV_over_rd'] = args.dv_col

    if args.debug_cols:
        print("CSV columns:", list(df.columns))
        print("Detected/overrides:", cols)
        return

    # grid
    if args.hELG_fixed is not None and args.hLRG2_fixed is not None:
        grid = [(args.hELG_fixed, args.hLRG2_fixed)]
    else:
        hv1 = np.linspace(args.hELG_range[0],  args.hELG_range[1],  int(args.hELG_range[2]))
        hv2 = np.linspace(args.hLRG2_range[0], args.hLRG2_range[1], int(args.hLRG2_range[2]))
        grid = [(a,b) for a in hv1 for b in hv2]

    rows = []
    best = None
    for hE, hL in grid:
        chi, dof, cd = chi2_with_scales(df, cov, cols, args.om0, args.h, hE, hL, args.elg_pattern, args.lrg2_pattern)
        row = dict(hELG=hE, hLRG2=hL, chi2=chi, dof=dof, chi2_dof=cd)
        rows.append(row)
        if best is None or chi < best['chi2']:
            best = row

    out_csv = f"{args.out_prefix}_summary.csv"
    pd.DataFrame(rows).sort_values('chi2').to_csv(out_csv, index=False)
    print(f"Best: hELG={best['hELG']:.6f}  hLRG2={best['hLRG2']:.6f}  chi2={best['chi2']:.3f}  dof={best['dof']}  chi2/dof={best['chi2_dof']:.3f}")
    print(f"CSV saved at: {out_csv}")

if __name__ == "__main__":
    main()
