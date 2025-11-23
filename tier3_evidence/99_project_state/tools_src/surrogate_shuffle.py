#!/usr/bin/env python3
import argparse, json, numpy as np
from _spectral_utils import load_residual_series, periodogram, fit_lorentz

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=None)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", required=True)
ap.add_argument("--window", default="hann")
ap.add_argument("--pad", type=int, default=4)
args = ap.parse_args()

x, r = load_residual_series(args.inp)
rng = np.random.default_rng(12345)
f_real, P_real = periodogram(x, r, pad_mul=args.pad, window=args.window)
fit_real = fit_lorentz(f_real, P_real)
real_peak = float(f_real[np.argmax(P_real)])

peaks = []
for i in range(args.n):
    # phase randomize in Fourier domain
    y = r - r.mean()
    n = y.size
    Y = np.fft.rfft(y)
    phases = rng.uniform(0, 2*np.pi, size=Y.size)
    Ys = np.abs(Y) * np.exp(1j*phases)
    ys = np.fft.irfft(Ys, n=n)
    f, P = periodogram(x, ys, pad_mul=args.pad, window=args.window)
    fit = fit_lorentz(f, P)
    peaks.append({"i": i, "f_peak": float(f[np.argmax(P)]), "Q": float(fit.x0/(2*fit.gamma))})
out = {
    "real": {"f_peak": real_peak, "Q": float(fit_real.x0/(2*fit_real.gamma))},
    "surrogates": peaks
}
with open(args.out, "w") as f: json.dump(out, f, indent=2)
print("Wrote", args.out, "surrogates=", len(peaks))
