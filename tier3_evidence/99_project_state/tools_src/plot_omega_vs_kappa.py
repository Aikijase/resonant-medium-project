#!/usr/bin/env python3
import json, re, glob
import numpy as np, matplotlib.pyplot as plt

pairs=[]
for path in sorted(glob.glob("outputs/phase8/kap_thresh_tau0p30_kap*_*summary.json")):
    J=json.load(open(path))
    # recover kappa from filename
    m=re.search(r"kap([-+]\d+\.\d+)", path)
    if not m: continue
    k=float(m.group(1))
    w= J.get("lorentzian",{}).get("omega0") or J.get("quad_peak_omega") or J.get("omega_at_max")
    pairs.append((k, float(w)))

pairs.sort()
ks=[k for k,_ in pairs]; ws=[w for _,w in pairs]
plt.figure(figsize=(7,4))
plt.plot(ks, ws, marker="o")
plt.axhline(0.55, lw=1, alpha=0.4)  # guide
plt.axvline(-0.02, lw=1, ls="--", alpha=0.4); plt.axvline(0.02, lw=1, ls="--", alpha=0.4)
plt.xlabel("κ"); plt.ylabel("ω*"); plt.title("τ=0.30: dominant frequency vs κ")
plt.tight_layout(); plt.savefig("outputs/phase8/omega_vs_kappa_tau0p30.png", dpi=150)
print("[wrote] outputs/phase8/omega_vs_kappa_tau0p30.png")
