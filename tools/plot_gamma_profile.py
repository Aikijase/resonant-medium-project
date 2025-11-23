#!/usr/bin/env python3
import json, glob, os, re, math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("outputs")
PATTERN = str(OUTDIR / "joint_gamma_fix_*.json")
CSV_OUT = OUTDIR / "gamma_profile.csv"
PNG_OUT = OUTDIR / "gamma_profile.png"

def extract_gamma(J, fname):
    # Try JSON fields first
    for path in [("best_params","gamma"), ("priors","gamma","mean"), ("fit","gamma_value")]:
        d=J
        try:
            for k in path: d=d[k]
            if isinstance(d,(int,float)): return float(d)
        except Exception:
            pass
    # Fallback: parse from filename
    m = re.search(r"joint_gamma_fix_([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)", os.path.basename(fname))
    return float(m.group(1).replace("p",".")) if m else None

rows = []
for path in sorted(glob.glob(PATTERN)):
    try:
        with open(path) as f:
            J = json.load(f)
        g = extract_gamma(J, path)
        chi = J.get("chi2", None)
        if g is None or chi is None: 
            continue
        rows.append((float(g), float(chi), os.path.basename(path)))
    except Exception as e:
        print(f"[warn] skip {path}: {e}")

if not rows:
    raise SystemExit("No usable profile files found.")

# Deduplicate by gamma → keep min chi2
byg = {}
for g,chi,b in rows:
    if (g not in byg) or (chi < byg[g][0]):
        byg[g] = (chi, b)
gam = np.array(sorted(byg.keys()))
chi = np.array([byg[g][0] for g in gam])

chi_min = float(chi.min())
g_best = float(gam[chi.argmin()])
dchi = chi - chi_min

def ci_from_profile(g, d, level):
    # piecewise-linear on each side; return (L,R) or (nan,nan) if not bracketed
    L = R = math.nan
    # left side
    mask = g <= g_best
    G, D = g[mask], d[mask]
    for i in range(len(G)-1, 0, -1): # from best downward
        g0,g1 = G[i-1], G[i]; d0,d1 = D[i-1], D[i]
        if (d0-level)*(d1-level) <= 0 and g0 != g1:
            t = (level - d0) / (d1 - d0) if d1 != d0 else 0.0
            L = g0 + t*(g1 - g0)
            break
    # right side
    mask = g >= g_best
    G, D = g[mask], d[mask]
    for i in range(0, len(G)-1):
        g0,g1 = G[i], G[i+1]; d0,d1 = D[i], D[i+1]
        if (d0-level)*(d1-level) <= 0 and g0 != g1:
            t = (level - d0) / (d1 - d0) if d1 != d0 else 0.0
            R = g0 + t*(g1 - g0)
            break
    return L, R

g68 = ci_from_profile(gam, dchi, 1.00)
g95 = ci_from_profile(gam, dchi, 3.84)

# CSV
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_OUT, "w") as f:
    f.write("gamma,chi2,delta_chi2\n")
    for g,c,dc in zip(gam, chi, dchi):
        f.write(f"{g:.6f},{c:.6f},{dc:.6f}\n")

def fmt(iv):
    L,R = iv
    return "[n/a, n/a]" if any(map(lambda x: (x is None) or (isinstance(x,float) and math.isnan(x)), (L,R))) else f"[{L:.3f}, {R:.3f}]"

print(f"best gamma = {g_best:.3f}")
print(f"chi2_min   = {chi_min:.3f}")
print(f"68% CI     = {fmt(g68)}  (Δχ²=1)")
print(f"95% CI     = {fmt(g95)}  (Δχ²=3.84)")

plt.figure(figsize=(6,4.2), dpi=140)
plt.plot(gam, dchi, marker="o")
plt.axhline(1.0, ls="--"); plt.axhline(3.84, ls="--"); plt.axvline(g_best, ls=":")
plt.xlabel(r"$\gamma$"); plt.ylabel(r"$\Delta\chi^2(\gamma)$"); plt.title(r"Profile $\Delta\chi^2$ vs $\gamma$")
if not (math.isnan(g68[0]) or math.isnan(g68[1])): plt.plot([g68[0], g68[1]], [1.0, 1.0], marker="|")
if not (math.isnan(g95[0]) or math.isnan(g95[1])): plt.plot([g95[0], g95[1]], [3.84, 3.84], marker="|")
plt.tight_layout(); plt.savefig(PNG_OUT)
print(f"Wrote {CSV_OUT}\nWrote {PNG_OUT}")
