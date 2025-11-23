#!/usr/bin/env python3
"""
Phase-6: Resonant Response Curve
Combines fs8 (Phase-2), g×κ (Phase-4), and damping Γ(z) (Phase-5)
to measure the redshift-dependent resonant response R(z).
"""

import json, csv, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

FS8_CSV  = "outputs/phase2/fs8_eval.csv"
GK_MULTI = "outputs/phase4/gk_multi.json"
DAMP_SUM = "outputs/phase5/damping_runs.json"
OUT_CSV  = "outputs/phase6/resonant_response.csv"
OUT_JSON = "outputs/phase6/resonant_response.json"
OUT_PNG  = "plots/phase6_resonant_response.png"

Path("outputs/phase6").mkdir(parents=True, exist_ok=True)
Path("plots/phase6").mkdir(parents=True, exist_ok=True)

# --- Load ---
fs8 = pd.read_csv(FS8_CSV)
gk  = json.load(open(GK_MULTI))
damp = json.load(open(DAMP_SUM))

# Effective damping by redshift (interpolate Γ vs z)
z_damp = np.linspace(0,2,len(damp["runs"]))
gamma  = np.array([r.get("gamma_eff",0.0) for r in damp["runs"]])
Γ = np.interp(fs8["z"], z_damp, gamma, left=gamma[0], right=gamma[-1])

# Get lensing amplitude ratio A_hat/A_pred
A_pred = gk["scale_model"]
A_hat  = gk["A_hat"]
A_ratio = A_hat / A_pred if A_pred != 0 else 1.0

# Compute R(z)
R = (fs8["fs8_fit"] / fs8["fs8_pred"]) * A_ratio * np.exp(-Γ)
fs8["R_res"] = R
fs8["Gamma"] = Γ

# --- Save ---
fs8.to_csv(OUT_CSV, index=False)
json.dump({"A_ratio":A_ratio,"R_stats":{"mean":float(np.nanmean(R)),"std":float(np.nanstd(R))}}, open(OUT_JSON,"w"), indent=2)
print(f"Wrote {OUT_CSV} and {OUT_JSON}")

# --- Plot ---
plt.figure(figsize=(6,4))
plt.axhline(1, color='k', ls='--', lw=1)
plt.plot(fs8["z"], R, "o-", label="Resonant Response R(z)")
plt.xlabel("Redshift z")
plt.ylabel("R(z)")
plt.title("Phase-6: Resonant Response Curve")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"Wrote {OUT_PNG}")
