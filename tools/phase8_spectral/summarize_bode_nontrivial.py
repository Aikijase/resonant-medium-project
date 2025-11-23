#!/usr/bin/env python3
"""
Summarize non-DC spectral peaks across (tau, kappa)
---------------------------------------------------
Reads all *_bode.csv files, skips the first few low-ω bins,
finds the dominant residual-power peak, and builds heatmaps
for ω* and its power.

Change K_SKIP to control how many bins to ignore.
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import re

BASE = Path("outputs/phase8")
K_SKIP = 3        # ignore first N ultra-low-freq bins
W_MAX  = None     # set to e.g. 5.0 to cap freq range

rows=[]
# Match files like: sweep_tau0.10_kap+0.00_bode.csv  (no extra underscore before _bode)
for bode in sorted(BASE.glob("sweep_tau*_kap*_bode.csv")):
    m = re.search(r"sweep_tau([0-9.]+)_kap([+\-][0-9.]+)_bode\.csv$", bode.name)
    if not m:
        continue
    tau, kappa = float(m.group(1)), float(m.group(2))
    df = pd.read_csv(bode)
    if W_MAX is not None:
        df = df[df["omega"] <= W_MAX]
    if len(df) <= K_SKIP:
        continue
    df2 = df.iloc[K_SKIP:].copy()
    i = int(df2["P_residual"].idxmax())
    w = float(df2.loc[i, "omega"])
    P = float(df2.loc[i, "P_residual"])
    rows.append({"tau":tau, "kappa":kappa, "omega_peak":w, "power_peak":P})

if not rows:
    print("No bode files found.")
    raise SystemExit

df = pd.DataFrame(rows)
csv_out = BASE / "sweep_tau_kappa_bode_nontrivial.csv"
df.to_csv(csv_out, index=False)
print("Wrote", csv_out)

# --- Heatmaps ---
def heat(col, title, cmap="viridis"):
    pt = df.pivot(index="tau", columns="kappa", values=col).sort_index().sort_index(axis=1)
    plt.figure(figsize=(6,4))
    im = plt.imshow(pt.values, origin="lower", aspect="auto", cmap=cmap)
    plt.xticks(range(len(pt.columns)), [f"{c:+.2f}" for c in pt.columns])
    plt.yticks(range(len(pt.index)),   [f"{r:.2f}"  for r in pt.index])
    plt.xlabel("kappa"); plt.ylabel("tau")
    plt.title(title)
    plt.colorbar(im,label=col)
    out = csv_out.with_name(csv_out.stem + f"_{col}_heatmap.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print("Wrote", out)

heat("omega_peak", "Non-DC ω* (skip first bins)")
heat("power_peak", "Peak Power (skip first bins)", cmap="magma")
