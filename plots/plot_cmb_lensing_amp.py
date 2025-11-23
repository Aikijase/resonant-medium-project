#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt

J = json.load(open("outputs/phase4/lensing_score.json"))
rows = J["rows"]

labels = [r["survey"] for r in rows]
x = np.arange(len(rows))
Amod = [r["A_model"] for r in rows]
Aobs = [r["A_L"] for r in rows]
Sobs = [r["sigma_A"] for r in rows]

plt.figure()
plt.errorbar(x, Aobs, yerr=Sobs, fmt='o', label='Observed $A_L$')
plt.plot(x, Amod, 's', label='Model $A_L$')
plt.axhline(1.0, linestyle='--', label='LCDM $A_L=1$')
plt.xticks(x, labels, rotation=15, ha='right')
plt.ylabel('CMB lensing amplitude $A_L$')
plt.title('CMB lensing: model vs observed')
plt.tight_layout()
plt.savefig('plots/cmb_lensing_amp.png', dpi=150)
print("Wrote plots/cmb_lensing_amp.png")
