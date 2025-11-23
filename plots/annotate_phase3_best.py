import json, pandas as pd, numpy as np, matplotlib.pyplot as plt

D = pd.read_csv("outputs/phase3/fs8_gammaevo_grid.csv")
B = json.load(open("outputs/phase3/fs8_gammaevo_best.json"))["best"]
g0_best, g1_best, WWI_best = float(B["gamma0"]), float(B["g1"]), B.get("WWI")

g0 = np.sort(D["gamma0"].unique())
g1 = np.sort(D["g1"].unique())

def gridize(col):
    M = np.full((len(g1), len(g0)), np.nan)
    for i,gi in enumerate(g1):
        for j,g0i in enumerate(g0):
            v = D[(D["gamma0"]==g0i)&(D["g1"]==gi)][col]
            if len(v): M[i,j] = v.values[0]
    return M

M = gridize("WWI")
plt.figure(figsize=(7.2,5.6))
im = plt.imshow(M, origin="lower", aspect="auto",
                extent=[g0.min(), g0.max(), g1.min(), g1.max()])
plt.colorbar(im, label="WWI (%) vs ΛCDM (fs8)")
CS = plt.contour(M, levels=[50,70,85], colors="white",
                 extent=[g0.min(), g0.max(), g1.min(), g1.max()])
plt.clabel(CS, inline=True, fmt="WWI=%d", fontsize=8)

# Mark the best point
plt.plot([g0_best], [g1_best], marker='o', markersize=6)
label = f"best (γ₀={g0_best:.3f}, g₁={g1_best:.3f}, WWI={WWI_best})"
plt.text(g0_best, g1_best, "  " + label, va="center", ha="left")

plt.xlabel("γ₀")
plt.ylabel("g₁  (γ(z)=γ₀ + g₁ z/(1+z))")
plt.title("Phase-3: fs8 WWI across γ(z) — best point")
plt.tight_layout()
plt.savefig("plots/phase3_gammaevo_heatmap_annotated.png", dpi=150)
print("Wrote plots/phase3_gammaevo_heatmap_annotated.png")
