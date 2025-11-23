#!/usr/bin/env python3
import csv, math, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

path = Path("outputs/phase8/sweep_tau_kappa.csv")

def fnum(x):
    try:
        return float(x)
    except: return None

W = {}
with open(path) as fin:
    reader = csv.DictReader(fin)
    for row in reader:
        if row.get("status") != "ok": continue
        tau = fnum(row.get("tau")); kap = fnum(row.get("kappa"))
        w0  = fnum(row.get("lorentz_omega0") or row.get("omega_at_max"))
        if None in (tau, kap, w0): continue
        W.setdefault(kap, []).append((tau, w0))

for k in list(W.keys()):
    W[k].sort()

plt.figure(figsize=(7,4))
for k in sorted(W.keys()):
    ts = [t for t,_ in W[k]]
    ws = [w for _,w in W[k]]
    plt.plot(ts, ws, marker="o", label=f"κ={k:+.2f}")

plt.xlabel("τ"); plt.ylabel("ω*")
plt.title("Peak ω* vs τ (per κ)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
out="outputs/phase8/sweep_peak_lines.png"
plt.savefig(out, dpi=150); plt.close()
print(f"[wrote] {out}")
