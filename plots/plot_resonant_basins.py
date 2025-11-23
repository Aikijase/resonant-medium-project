import pandas as pd, numpy as np, matplotlib.pyplot as plt

DF = pd.read_csv("outputs/phase2/basin_grid.csv")

# Grids
fvals = np.sort(DF["f"].unique())
gvals = np.sort(DF["gamma"].unique())
def gridize(col):
    M = np.zeros((len(gvals), len(fvals)))
    for i,g in enumerate(gvals):
        for j,f in enumerate(fvals):
            v = DF[(DF["f"]==f) & (DF["gamma"]==g)][col]
            M[i,j] = v.values[0] if len(v) else np.nan
    return M

M_WWI    = gridize("WWI")
M_STABLE = gridize("stable")
M_STRESS = gridize("stress_relL2")

plt.figure(figsize=(7.2,5.6))
# base: WWI colormap
im = plt.imshow(M_WWI, origin="lower", aspect="auto",
                extent=[fvals.min(), fvals.max(), gvals.min(), gvals.max()])
cbar = plt.colorbar(im, label="WWI (%)")

# overlay: stability mask (hatching)
stable_mask = np.where(M_STABLE>0.5, 1.0, np.nan)
plt.contourf(np.linspace(fvals.min(),fvals.max(),M_WWI.shape[1]),
             np.linspace(gvals.min(),gvals.max(),M_WWI.shape[0]),
             stable_mask, levels=[0.5,1.5], hatches=['///'], alpha=0.0, colors='none')

# contours: WWI thresholds for “resonant basins”
levels = [50, 70, 85]  # adjust as you like
CS = plt.contour(M_WWI, levels=levels, colors="white",
                 extent=[fvals.min(), fvals.max(), gvals.min(), gvals.max()])
plt.clabel(CS, inline=True, fmt="WWI=%d", fontsize=8)

plt.xlabel("f"); plt.ylabel("γ")
plt.title("Resonant basins: WWI contours + stability hatch")
plt.tight_layout()
plt.savefig("plots/resonant_basins.png", dpi=150)
print("Wrote plots/resonant_basins.png")
