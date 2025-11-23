#!/usr/bin/env python3
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_rhode(path):
    t=pd.read_csv(path)
    zcol='z' if 'z' in t.columns else [c for c in t.columns if c.lower().startswith('z')][0]
    rhocols=[c for c in t.columns if 'rho_de' in c.lower() or (('rho' in c.lower()) and ('de' in c.lower()))]
    return t[[zcol, rhocols[0]]].rename(columns={zcol:'z', rhocols[0]:'rho_de'}).sort_values('z')

def load_rdot(path_fallback, rho_bh_csv):
    # prefer explicit rho_dot CSV if present; else finite-diff from rho_bh
    if os.path.exists(path_fallback):
        t=pd.read_csv(path_fallback)
        zcol='z' if 'z' in t.columns else [c for c in t.columns if c.lower().startswith('z')][0]
        rdcols=[c for c in t.columns if 'rho' in c.lower() and 'dot' in c.lower()]
        if rdcols:
            return t[[zcol, rdcols[0]]].rename(columns={zcol:'z', rdcols[0]:'rho_dot_bh'}).sort_values('z')
    bh=pd.read_csv(rho_bh_csv).sort_values([c for c in ['z','Z'] if c in pd.read_csv(rho_bh_csv).columns][0])
    zcol='z' if 'z' in bh.columns else [c for c in bh.columns if c.lower().startswith('z')][0]
    rhcol=[c for c in bh.columns if 'rho' in c.lower()][0]
    z=bh[zcol].values; rho=bh[rhcol].values
    drho_dz=np.gradient(rho, z)
    # convert dρ/dz to dρ/dt up to H(z) factor (monotonic test OK for timing)
    return pd.DataFrame({'z':z, 'rho_dot_bh': -drho_dz})  # sign chosen so peak aligns visually with DE rise

def xcorr_lag(x, y, lags):
    # returns lag* with max Pearson r
    best=(None,-9)
    for L in lags:
        # shift y by L in z (positive L = compare x(z) with y(z+L))
        z = x['z'].values
        yr = np.interp(z, y['z'].values+L, y[y.columns[1]].values, left=np.nan, right=np.nan)
        mask = np.isfinite(yr)
        if mask.sum()<5: continue
        r = np.corrcoef(x[x.columns[1]].values[mask], yr[mask])[0,1]
        if np.isfinite(r) and r>best[1]: best=(L,r)
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rho-de', required=True, help='CSV with z, rho_de')
    ap.add_argument('--rho-dot', default='data/bh_accretion_from_lf.csv', help='Optional explicit rho_dot(z)')
    ap.add_argument('--rho-bh', default='data/rho_bh_z.clean.csv', help='Fallback for finite-diff if rho_dot missing')
    ap.add_argument('--out-prefix', required=True)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    de = load_rhode(args.rho_de)
    rdot = load_rdot(args.rho_dot, args.rho_bh)

    # use d rho_de / dz as a "rise" proxy
    dDe_dz = np.gradient(de['rho_de'].values, de['z'].values)
    dep = pd.DataFrame({'z':de['z'].values, 'de_slope': dDe_dz})

    lags = np.linspace(-1.5, 1.5, 121)  # Δz lead/lag
    best = xcorr_lag(rdot.rename(columns={'rho_dot_bh':'x'}), dep.rename(columns={'de_slope':'y'}), lags)
    Lbest, rbest = best

    # Save report
    rep={'lag_z_best': float(Lbest) if Lbest is not None else None, 'pearson_r': float(rbest)}
    with open(args.out_prefix+'.report.json','w') as f: json.dump(rep,f,indent=2)

    # Plot overlay at best lag
    z = de['z'].values
    de_s = dep['de_slope'].values
    rdot_shift = np.interp(z, rdot['z'].values+ (Lbest or 0.0), rdot['rho_dot_bh'].values, left=np.nan, right=np.nan)
    plt.figure()
    plt.plot(z, (rdot_shift/np.nanmax(np.abs(rdot_shift))), label=f"rho_dot_bh shifted by Δz={Lbest:.2f}")
    plt.plot(z, (de_s/np.nanmax(np.abs(de_s))), label="d rho_de / dz")
    plt.gca().invert_xaxis()
    plt.xlabel('z'); plt.ylabel('normalized')
    plt.legend()
    plt.savefig(args.out_prefix+'.plot.png', dpi=160, bbox_inches='tight')

    print(f"Timing test: best lag Δz={Lbest:.2f} with r={rbest:.2f}")
    print(f"Wrote {args.out_prefix}.report.json and {args.out_prefix}.plot.png")

if __name__ == '__main__':
    main()
