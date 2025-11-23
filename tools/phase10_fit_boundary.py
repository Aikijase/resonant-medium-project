#!/usr/bin/env python3
import csv, numpy as np, matplotlib.pyplot as plt, sys, math, pathlib

in_csv  = sys.argv[1] if len(sys.argv)>1 else "outputs/phase10/contour_from_map_fair.csv"
out_png = sys.argv[2] if len(sys.argv)>2 else "outputs/phase10/boundary_curve_fit.png"

# --- load contour ---
xs, ys = [], []
with open(in_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        k = row["Kphi_min_at_thr"]
        if k and k != "NA":
            xs.append(float(row["omega2"]))
            ys.append(float(k))

if not xs:
    print("No contour points available.")
    sys.exit(1)

xs = np.array(xs)
ys = np.array(ys)

# --- quadratic fit: y = a0 + a1 (x-x0) + a2 (x-x0)^2
x0 = xs.mean()  # centering helps numerics
X = np.vstack([np.ones_like(xs), (xs-x0), (xs-x0)**2]).T
coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
a0, a1, a2 = coef
xmin = x0 - a1/(2*a2) if a2 != 0 else x0
ymin = a0 - a1**2/(4*a2) if a2 != 0 else a0

# --- plot ---
xx = np.linspace(xs.min(), xs.max(), 200)
yy = a0 + a1*(xx-x0) + a2*(xx-x0)**2

plt.figure()
plt.plot(xs, ys, "o", label="data")
plt.plot(xx, yy, "-", label="quadratic fit")
plt.axvline(xmin, ls="--", lw=1, label=f"min @ ω₂={xmin:.3f}")
plt.xlabel(r"$\omega_2$")
plt.ylabel(r"$K_\phi^{\min}$ (sync ≥ threshold)")
plt.title("Boundary curve & quadratic fit")
plt.grid(True)
plt.legend()
plt.tight_layout()
pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=200)

print(f"Saved: {out_png}")
print(f"Fit: Kφ_min(ω₂) ≈ {a0:.4f} + {a1:.4f}(ω₂−{x0:.3f}) + {a2:.4f}(ω₂−{x0:.3f})²")
print(f"Estimated minimum at ω₂ ≈ {xmin:.4f}, Kφ_min ≈ {ymin:.4f}")
