#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, json

def fit_g_diag(overlay_csv, gmin=0.5, gmax=3.0, ng=251, drop_zero_sigma=True):
    df = pd.read_csv(overlay_csv)
    # columns expected from your overlay: z, DV_rel, DV_rel_sigma, pred_rel_spacing
    m = df['pred_rel_spacing'].to_numpy(float)
    y = df['DV_rel'].to_numpy(float)
    s = df['DV_rel_sigma'].to_numpy(float)

    mask = np.isfinite(y) & np.isfinite(m) & np.isfinite(s)
    if drop_zero_sigma:
        mask &= (s > 0)
    y, m, s = y[mask], m[mask], s[mask]
    if y.size < 3:
        raise SystemExit(f"Not enough usable points after masking (got {y.size}).")

    w = 1.0 / np.maximum(s, 1e-12)**2
    dof = max(len(y) - 2, 1)

    # Baseline g=1
    num = np.sum(w * m * y)
    den = np.sum(w * m * m)
    A0  = float(num / max(den, 1e-30))
    chi2_g1 = float(np.sum((y - A0*m)**2 * w))

    # Grid over g
    best = dict(chi2=np.inf, g=None, A=None, dof=dof)
    gs = np.linspace(gmin, gmax, ng)
    for g in gs:
        mg  = np.power(np.maximum(m, 1e-300), g)
        num = np.sum(w * mg * y)
        den = np.sum(w * mg * mg)
        A   = float(num / max(den, 1e-30))
        chi2 = float(np.sum((y - A*mg)**2 * w))
        if chi2 < best['chi2']:
            best.update(chi2=chi2, g=float(g), A=A)

    best['chi2_per_dof']     = best['chi2'] / best['dof']
    best['chi2_g1']          = chi2_g1
    best['chi2_g1_per_dof']  = chi2_g1 / best['dof']
    best['N_used']           = int(y.size)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overlay', required=True, help='*_overlay.csv from bao_rel_fit_plus.py')
    ap.add_argument('--out', default='outputs/bao_fitg_diag_summary.json')
    ap.add_argument('--gmin', type=float, default=0.5)
    ap.add_argument('--gmax', type=float, default=3.0)
    ap.add_argument('--ng',   type=int,   default=251)
    args = ap.parse_args()
    res = fit_g_diag(args.overlay, args.gmin, args.gmax, args.ng)
    with open(args.out,'w') as f: json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
