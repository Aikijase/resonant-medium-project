import pandas as pd, numpy as np, matplotlib.pyplot as plt

B  = pd.read_csv("outputs/phase2/basin_grid.csv")
I  = pd.read_csv("outputs/phase2/isw_ok_grid.csv")

fvals = np.sort(B["f"].unique())
gvals = np.sort(B["gamma"].unique())

def gridize(df, col):
    M = np.full((len(gvals), len(fvals)), np.nan)
    for i,g in enumerate(gvals):
        for j,f in enumerate(fvals):
            v = df[(df["f"]==f)&(df["gamma"]==g)][col]
            if len(v): M[i,j] = v.values[0]
    return M

M_WWI    = gridize(B, "WWI")          # from fs8 scorer in basins grid
M_STABLE = gridize(B, "stable")
M_ISWOK  = gridize(I, "isw_ok")

plt.figure(figsize=(7.6,5.8))
# WWI as image
im = plt.imshow(M_WWI, origin="lower", aspect="auto",
                extent=[fvals.min(), fvals.max(), gvals.min(), gvals.max()])
cbar = plt.colorbar(im, label="WWI (%) from fσ8")

# Stability hatch
stab_mask = np.where(M_STABLE>0.5, 1.0, np.nan)
plt.contourf(np.linspace(fvals.min(), fvals.max(), M_WWI.shape[1]),
             np.linspace(gvals.min(), gvals.max(), M_WWI.shape[0]),
             stab_mask, levels=[0.5,1.5], hatches=['///'], alpha=0.0, colors='none')

# WWI contours
levels = [50, 70, 85]
CS = plt.contour(M_WWI, levels=levels, colors="white",
                 extent=[fvals.min(), fvals.max(), gvals.min(), gvals.max()])
plt.clabel(CS, inline=True, fmt="WWI=%d", fontsize=8)

# ISW-OK outline (thin)
isw_mask = np.where(M_ISWOK>0.5, 1.0, 0.0)
plt.contour(isw_mask, levels=[0.5], linewidths=1.5, colors="black",
            extent=[fvals.min(), fvals.max(), gvals.min(), gvals.max()])

plt.xlabel("f"); plt.ylabel("γ")
plt.title("Resonant basins: WWI (color) + stability hatch + ISW-OK outline")
plt.tight_layout()
plt.savefig("plots/resonant_basins_isw.png", dpi=150)
print("Wrote plots/resonant_basins_isw.png")
