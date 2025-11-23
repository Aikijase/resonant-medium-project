import json, matplotlib.pyplot as plt, numpy as np
J=json.load(open("outputs/phase2/isw_score.json"))
rows=J["rows"]
labs=[r["survey"] for r in rows]
x=np.arange(len(rows))
Amod=[r["A_model"] for r in rows]
Aobs=[r["A_ISW"] for r in rows]
Sobs=[r["sigma_A"] for r in rows]

plt.figure()
plt.errorbar(x, Aobs, yerr=Sobs, fmt='o', label='Observed A')
plt.plot(x, Amod, 's', label='Model A')
plt.axhline(1.0, linestyle='--', label='LCDM A=1')
plt.xticks(x, labs, rotation=30, ha='right')
plt.ylabel('ISW amplitude A')
plt.title('ISW amplitude: model vs observed')
plt.tight_layout()
plt.savefig('plots/isw_amplitude.png', dpi=150)
print("Wrote plots/isw_amplitude.png")
