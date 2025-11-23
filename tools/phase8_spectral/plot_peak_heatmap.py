#!/usr/bin/env python3
import csv, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import math, sys

path = Path("outputs/phase8/sweep_tau_kappa.csv")
if not path.exists():
    print(f"missing {path}"); sys.exit(1)

def fnum(x):
    try:
        if x in ("", None, "None"): return math.nan
        return float(x)
    except: return math.nan

rows = []
with open(path) as f:
    r = csv.DictReader(f)
    for row in r:
        if row.get("status") != "ok": continue
        tau   = fnum(row.get("tau"))
        kap   = fnum(row.get("kappa"))
        w0    = fnum(row.get("lorentz_omega0")) if row.get("lorentz_omega0") else fnum(row.get("omega_at_max"))
        if math.isnan(tau) or math.isnan(kap) or math.isnan(w0): continue
        rows.append((tau, kap, w0))

taus   = sorted(set(t for t,_,_ in rows))
kappas = sorted(set(k for _,k,_ in rows))
M = np.full((len(taus), len(kappas)), np.nan)
for t,k,w in rows:
    i = taus.index(t); j = kappas.index(k)
    M[i,j] = w

fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(M, origin="lower", aspect="auto",
               extent=[min(kappas), max(kappas), min(taus), max(taus)])
c = plt.colorbar(im, ax=ax); c.set_label("ω*")
ax.set_xlabel("κ"); ax.set_ylabel("τ"); ax.set_title("Peak ω* over (τ, κ)")
out = "outputs/phase8/sweep_peak_heatmap.png"
plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
print(f"[wrote] {out}")
