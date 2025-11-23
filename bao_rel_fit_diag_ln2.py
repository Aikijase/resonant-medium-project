#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, json

def rel_spacing_ln2(z, a, b, O0, O1, O2, zref):
    x = np.log1p(z)
    Om = O0 + O1*x + O2*(x*x)
    k  = a*Om + b
    s  = 1.0/np.where(k>0, k, np.nan)
    # normalize at reference redshift
    xref = np.log1p(zref); Omr = O0 + O1*xref + O2*(xref*xref)
    kr = a*Omr + b; sr = 1.0/kr
    return s/sr

def fit_diag(overlay_csv, a, b, zref=None):
    df = pd.read_csv(overlay_csv)
    z  = df['z'].to_numpy(float)
    y  = df['DV_rel'].to_numpy(float)
    s  = df['DV_rel_sigma'].to_numpy(float)
    m  = np.isfinite(z)&np.isfinite(y)&np.isfinite(s)
    z,y,s = z[m],y[m],s[m]
    # drop the reference row with zero sigma if present
    m2 = s>0; z,y,s = z[m2],y[m2],s[m2]
    if zref is None:
        # pick the minimum z in the retained sample as reference
        zref = float(np.min(z))
    w = 1.0/np.maximum(s,1e-12)**2
    best = dict(chi2=np.inf)
    # coarse but quick grid (feel free to widen)
    O0s = np.linspace(0.6, 1.6, 51)
    O1s = np.linspace(-1.5, 2.0, 71)
    O2s = np.linspace(-1.5, 2.0, 71)
    for O0 in O0s:
        for O1 in O1s[::2]:      # stride 2 to keep this snappy
            for O2 in O2s[::2]:
                mrel = rel_spacing_ln2(z, a,b, O0,O1,O2, zref)
                # analytic amplitude A in diagonal WLS
                num = np.sum(w*mrel*y)
                den = np.sum(w*mrel*mrel)
                A   = float(num/max(den,1e-30))
                chi2= float(np.sum((y-A*mrel)**2 * w))
                if chi2 < best['chi2']:
                    best = dict(O0=float(O0), O1=float(O1), O2=float(O2),
                                A=A, chi2=chi2, dof=max(len(z)-3-1,1), zref=zref)
    best['chi2_per_dof'] = best['chi2']/best['dof']
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--overlay', required=True)
    ap.add_argument('--a', type=float, default=0.078)
    ap.add_argument('--b', type=float, default=-0.011)
    ap.add_argument('--out', default='outputs/bao_diag_ln2_summary.json')
    args=ap.parse_args()
    res=fit_diag(args.overlay, args.a, args.b)
    with open(args.out,'w') as f: json.dump(res,f,indent=2)
    print(json.dumps(res,indent=2))
if __name__=='__main__': main()
