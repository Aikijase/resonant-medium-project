import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import tomllib
import pandas as pd
import matplotlib.pyplot as plt
from growth import solve_growth

with open("configs/resonant.toml", "rb") as fh:
    P = tomllib.load(fh)["params"]

res = solve_growth(P)
z_pred = res["z"]
fs8_pred = res["fs8"]

D = pd.read_csv("data/growth/fs8_catalog.csv")

plt.figure()
plt.errorbar(D["z"], D["fs8_obs"], yerr=D["sigma"], fmt="o", label="fσ8 data")
plt.plot(z_pred, fs8_pred, label="Resonant prediction")
plt.xlabel("z"); plt.ylabel("fσ8")
plt.legend(); plt.title("fσ8: data vs resonant prediction")
plt.tight_layout()
plt.savefig("plots/fs8_overlay.png", dpi=150)
print("Wrote plots/fs8_overlay.png")
