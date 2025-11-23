#!/usr/bin/env python3
import argparse, numpy as np, json
from _spectral_utils import load_residual_series, periodogram, fit_lorentz

def iaaft(y, iters=50, rng=None):
    if rng is None: rng = np.random.default_rng()
    y = np.asarray(y); n = y.size
    y_sorted = np.sort(y)
    # target amplitude spectrum
    Y = np.fft.rfft(y); amp = np.abs(Y)
    x = rng.permutation(y)  # start from shuffled
    for _ in range(iters):
        # enforce target spectrum
        X = np.fft.rfft(x); X = amp * np.exp(1j*np.angle(X))
        x = np.fft.irfft(X, n=n)
        # enforce amplitude distribution
        ranks = x.argsort().argsort()
        x = y_sorted[ranks]
    return x

ap=argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=None)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--out", required=True)
ap.add_argument("--window", default="hann")
ap.add_argument("--pad", type=int, default=4)
a=ap.parse_args()

x,r = load_residual_series(a.inp)
rng = np.random.default_rng(2025)
f_real,P_real = periodogram(x, r, pad_mul=a.pad, window=a.window)
fit_real = fit_lorentz(f_real, P_real)
real = {"f_peak": float(f_real[np.argmax(P_real)]),
        "Q": float(fit_real.x0/(2*fit_real.gamma))}

sur=[]
for i in range(a.n):
    ys = iaaft(r, a.iters, rng)
    f, P = periodogram(x, ys, pad_mul=a.pad, window=a.window)
    fit = fit_lorentz(f, P)
    sur.append({"f_peak": float(f[np.argmax(P)]),
                "Q": float(fit.x0/(2*fit.gamma))})

json.dump({"real": real, "surrogates": sur}, open(a.out,"w"), indent=2)
print("Wrote", a.out, "surrogates=",len(sur))
