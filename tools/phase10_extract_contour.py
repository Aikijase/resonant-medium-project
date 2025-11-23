#!/usr/bin/env python3
import csv, math, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

in_csv  = sys.argv[1] if len(sys.argv)>1 else "outputs/phase10/syncmap_band_fast.csv"
out_csv = sys.argv[2] if len(sys.argv)>2 else "outputs/phase10/contour_from_map.csv"
out_png = sys.argv[3] if len(sys.argv)>3 else "outputs/phase10/syncmap_band_fast_with_contour.png"
thr     = float(sys.argv[4]) if len(sys.argv)>4 else 0.95  # threshold

# --- load grid ---
with open(in_csv) as f:
    r = csv.reader(f)
    header = next(r)
    assert header[0] == "omega2", "unexpected CSV header"
    ks = [float(h.split("=",1)[1]) for h in header[1:]]
    omega2 = []
    Mrows = []
    for row in r:
        omega2.append(float(row[0]))
        vals = [float(x) if (x and x!="NaN") else np.nan for x in row[1:]]
        Mrows.append(vals)
omega2 = np.array(omega2)
ks = np.array(ks)
M = np.array(Mrows, dtype=float)  # shape (nw, nk)

# --- extract Kphi_min per omega2 by finding first crossing to >= thr ---
cont_w, cont_k = [], []
for i, w2 in enumerate(omega2):
    row = M[i]
    # Find the first index j where value >= thr (skips NaNs)
    j0 = None
    for j in range(len(ks)):
        s = row[j]
        if not math.isnan(s) and s >= thr:
            j0 = j; break
    if j0 is None:
        continue  # no lock in this row

    if j0 == 0:
        # already above threshold at first bin; take that bin as boundary
        k_star = ks[0]
    else:
        s_lo, s_hi = row[j0-1], row[j0]
        k_lo, k_hi = ks[j0-1], ks[j0]
        if not (math.isnan(s_lo) or math.isnan(s_hi)) and s_hi != s_lo:
            alpha = (thr - s_lo) / (s_hi - s_lo)
            k_star = k_lo + alpha * (k_hi - k_lo)
        else:
            k_star = ks[j0]
    cont_w.append(w2)
    cont_k.append(k_star)

# --- write contour csv ---
pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["omega2", "Kphi_min_at_thr"])
    for w2, k in zip(cont_w, cont_k):
        w.writerow([f"{w2:.6f}", f"{k:.6f}"])

print(f"wrote: {out_csv}")

# --- plot heatmap + contour overlay ---
plt.figure()
# clip for contrast, re-use the same extent as your map
v = np.clip(M, thr, 1.0)
plt.imshow(v.T, origin="lower", aspect="auto",
           extent=[omega2[0], omega2[-1], ks[0], ks[-1]])
plt.colorbar(label="sync_index (clipped ≥ threshold)")
if cont_w:
    plt.plot(cont_w, cont_k, "o-", linewidth=1.5, markersize=4, label=f"Contour @ {thr}")
plt.xlabel(r"$\omega_2$")
plt.ylabel(r"$K_\phi$")
plt.title("Phase-coupling sync map with ≥thr contour")
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print(f"wrote: {out_png}")
