#!/usr/bin/env python3
import argparse, numpy as np, json
from _spectral_utils import load_residual_series, fit_lorentz
from scipy.signal.windows import dpss

ap=argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=None)
ap.add_argument("--nw", type=float, default=3.5, help="time-halfbandwidth")
ap.add_argument("--K", type=int, default=4, help="# of tapers")
ap.add_argument("--out", required=True)
a=ap.parse_args()

x,r = load_residual_series(a.inp)
y = r - r.mean()
n = y.size
d = (x[1]-x[0]) if x.size>1 else 1.0
tapers = dpss(n, a.nw, Kmax=a.K, return_ratios=False)
specs=[]; freqs=None
for v in tapers:
    Y = np.fft.rfft(y*v)
    f = np.fft.rfftfreq(n, d=d)
    P = (np.abs(Y)**2)/n
    freqs = f
    specs.append(P)
S = np.mean(specs, axis=0)
fit = fit_lorentz(freqs, S)
Q = fit.x0/(2*fit.gamma) if fit.gamma>0 else 0.0
json.dump({"lorentz_f0": float(fit.x0), "Q": float(Q)}, open(a.out,"w"), indent=2)
print("Wrote", a.out, "f0=",float(fit.x0),"Q=",float(Q))
