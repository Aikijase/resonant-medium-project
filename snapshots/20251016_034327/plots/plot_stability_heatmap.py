import pandas as pd, numpy as np, matplotlib.pyplot as plt

DF = pd.read_csv("outputs/phase2/stability_map.csv")

# Score: higher when stress and total variation are lower; zero if not stable
score = np.where(DF["stable"], 1.0 / (1e-6 + DF["stress_relL2"] + 0.5*DF["TV_D"]), 0.0)

# Pivot for imshow
fvals = np.sort(DF["f"].unique())
gvals = np.sort(DF["gamma"].unique())
M = np.zeros((len(gvals), len(fvals)))
for i, g in enumerate(gvals):
    for j, f in enumerate(fvals):
        s = DF[(DF["f"]==f) & (DF["gamma"]==g)]
        M[i, j] = score[s.index[0]] if len(s) else 0.0

plt.figure()
im = plt.imshow(M, origin="lower", aspect="auto",
                extent=[fvals.min(), fvals.max(), gvals.min(), gvals.max()])
plt.colorbar(im, label="Stability score (↑ better)")
plt.contour(np.where(M>0,1,0), levels=[0.5], colors="white", linewidths=1,
            extent=[fvals.min(), fvals.max(), gvals.min(), gvals.max()])
plt.xlabel("f"); plt.ylabel("γ")
plt.title("Resonant Stability ≍ Damping (stress test)")
plt.tight_layout()
plt.savefig("plots/stability_heatmap.png", dpi=150)
print("Wrote plots/stability_heatmap.png")
