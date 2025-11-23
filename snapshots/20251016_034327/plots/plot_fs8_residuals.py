import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import tomllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from growth import solve_growth

with open("configs/resonant.toml", "rb") as fh:
    P = tomllib.load(fh)["params"]

res = solve_growth(P)
z_pred = np.asarray(res["z"])
fs8_pred = np.asarray(res["fs8"])

D = pd.read_csv("data/growth/fs8_catalog.csv")
fs8_at_data = np.interp(D["z"].values, z_pred, fs8_pred)
resid = D["fs8_obs"].values - fs8_at_data

plt.figure()
plt.axhline(0.0, linewidth=1)
plt.errorbar(D["z"], resid, yerr=D["sigma"], fmt="o")
plt.xlabel("z"); plt.ylabel("fσ8 (data − model)")
plt.title("fσ8 residuals")
plt.tight_layout()
plt.savefig("plots/fs8_residuals.png", dpi=150)
print("Wrote plots/fs8_residuals.png")
