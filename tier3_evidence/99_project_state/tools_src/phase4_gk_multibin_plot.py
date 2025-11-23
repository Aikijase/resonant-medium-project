#!/usr/bin/env python3
"""
Plot residuals (σ) per bin from outputs/phase4/gk_multi.json to plots/gk_multi_residuals.png
"""
import json, os
import matplotlib.pyplot as plt

J = json.load(open("outputs/phase4/gk_multi.json"))
labels = [f"{b['survey']}:{b['bin_id']}" for b in J["bins"]]
resids = []
for b in J["bins"]:
    # compute on the fly to avoid depending on scorer internals
    resids.append((b["A_lit"] - J["scale_model"]) / b["sigma_lit"])

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(9, 5))
plt.axhspan(-1, 1, alpha=0.1)
plt.axhline(0, linewidth=1)
plt.bar(range(len(labels)), resids)
plt.xticks(range(len(labels)), labels, rotation=25, ha='right')
plt.ylabel("Residual (σ)")
plt.title("g×κ residuals per bin (shape-only)")
plt.tight_layout()
plt.savefig("plots/gk_multi_residuals.png", dpi=160)
print("Wrote plots/gk_multi_residuals.png")
