#!/usr/bin/env python3
import argparse, json
from _spectral_utils import load_residual_series, periodogram, fit_lorentz

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=None, help="Residual series file (CSV/NPZ/JSON). Default tries outputs/phase8/*")
ap.add_argument("--pad", default="1,2,4,8")
ap.add_argument("--window", default="rect,hann,blackman")
ap.add_argument("--fit", default="lorentz", choices=["lorentz"])
ap.add_argument("--out", required=True)
args = ap.parse_args()

x, r = load_residual_series(args.inp)
pads = [int(p) for p in args.pad.split(",")]
wins = [w.strip() for w in args.window.split(",")]
rows = []
for p in pads:
    for w in wins:
        f, P = periodogram(x, r, pad_mul=p, window=w)
        # restrict to positive frequencies
        j = P.argmax()
        f0 = float(f[j]); P0 = float(P[j])
        fit = fit_lorentz(f, P)
        Q = fit.x0/(2*fit.gamma) if fit.gamma>0 else 0.0
        rows.append({
            "pad": p, "window": w,
            "f_peak": f0, "P_peak": P0,
            "lorentz_A": fit.A, "lorentz_f0": fit.x0, "lorentz_gamma": fit.gamma,
            "Q": Q, "chi2_fit": fit.chi2
        })
out = {"n": len(rows), "rows": rows}
with open(args.out, "w") as f: json.dump(out, f, indent=2)
print("Wrote", args.out, "rows=", len(rows))
