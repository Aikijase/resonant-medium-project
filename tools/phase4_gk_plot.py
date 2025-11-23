# tools/phase4_gk_plot.py
#!/usr/bin/env python3
"""
Bar plot of g×κ residuals (in σ) from outputs/phase4/gk_score.json.
Writes plots/gk_residuals.png
"""
import json, os
import matplotlib.pyplot as plt

J = json.load(open("outputs/phase4/gk_score.json"))
names  = [d["name"] for d in J["details"]]
resids = [d["resid_sigma_model"] for d in J["details"]]

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 4.5))
plt.axhspan(-1, 1, alpha=0.1)    # ±1σ band
plt.axhline(0, linewidth=1)
plt.bar(range(len(names)), resids)
plt.xticks(range(len(names)), names, rotation=20, ha='right')
plt.ylabel("Residual (σ)")
plt.title("g×κ residuals (shape-only)")
plt.tight_layout()
outfile = "plots/gk_residuals.png"
plt.savefig(outfile, dpi=160)
print(f"Wrote {outfile}")
