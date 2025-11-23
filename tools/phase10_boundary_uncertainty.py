#!/usr/bin/env python3
import csv, sys, math, pathlib
import numpy as np
import matplotlib.pyplot as plt

mean_csv = sys.argv[1] if len(sys.argv)>1 else "outputs/phase10/syncmap_band_fair_seeds_mean.csv"
std_csv  = sys.argv[2] if len(sys.argv)>2 else "outputs/phase10/syncmap_band_fair_seeds_std.csv"
thr      = float(sys.argv[3]) if len(sys.argv)>3 else 0.95
out_png  = sys.argv[4] if len(sys.argv)>4 else "outputs/phase10/boundary_uncertainty.png"

def load_grid(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        ks = [float(h.split("=",1)[1].split()[0]) for h in header[1:]]
        w2s, rows = [], []
        for row in r:
            w2s.append(float(row[0]))
            rows.append([float(x) if x else float("nan") for x in row[1:]])
    return np.array(w2s), np.array(ks), np.array(rows, dtype=float)

def contour_at_threshold(w2s, ks, M, thr):
    cw, ck = [], []
    for i, w2 in enumerate(w2s):
        row = M[i]
        idx = None
        for j in range(1, len(ks)):
            s_lo, s_hi = row[j-1], row[j]
            if not (math.isnan(s_lo) or math.isnan(s_hi)) and s_lo < thr <= s_hi:
                idx = j; break
        if idx is None: 
            continue
        j = idx
        s_lo, s_hi = row[j-1], row[j]
        k_lo, k_hi = ks[j-1], ks[j]
        if s_hi != s_lo:
            alpha = (thr - s_lo) / (s_hi - s_lo)
            k_star = k_lo + alpha * (k_hi - k_lo)
        else:
            k_star = k_hi
        cw.append(w2); ck.append(k_star)
    return np.array(cw), np.array(ck)

w2m, ksm, Mmean = load_grid(mean_csv)
w2s, kss, Mstd  = load_grid(std_csv)
assert np.allclose(w2m, w2s) and np.allclose(ksm, kss)

w_mean, k_mean = contour_at_threshold(w2m, ksm, Mmean, thr)
w_lo,   k_lo   = contour_at_threshold(w2m, ksm, np.clip(Mmean - Mstd, -1, 1), thr)
w_hi,   k_hi   = contour_at_threshold(w2m, ksm, np.clip(Mmean + Mstd, -1, 1), thr)

pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.figure()
plt.plot(w_mean, k_mean, "-",  label=f"mean @ {thr}")
if len(w_lo): plt.plot(w_lo,   k_lo,   "--", label="mean - std")
if len(w_hi): plt.plot(w_hi,   k_hi,   "--", label="mean + std")
plt.xlabel(r"$\omega_2$")
plt.ylabel(r"$K_\phi^{\min}$")
plt.title("Boundary with uncertainty band (mean±std)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print(f"Saved: {out_png}")
