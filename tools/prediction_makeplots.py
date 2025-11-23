#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pred = root / "outputs" / "predictions"
pred.mkdir(parents=True, exist_ok=True)

# Plot 1: Lensing residual template
df1 = pd.read_csv(pred/"lensing_residual_template.csv")
plt.figure()
plt.plot(df1["ell"], df1["residual_template"])
plt.xlabel("Multipole l")
plt.ylabel("Residual template (arb.)")
plt.title("P1: Low-omega Lensing Residual (Template)")
plt.grid(True)
plt.tight_layout()
plt.savefig(pred/"plot_lensing_template.png", dpi=160)
plt.close()

# Plot 2: BAO phase tweak (proxy)
df2 = pd.read_csv(pred/"bao_phase_tweak_proxy.csv")
plt.figure()
plt.plot(df2["z"], df2["bao_base"], label="BAO base")
plt.plot(df2["z"], df2["with_memory"], label="with memory")
plt.plot(df2["z"], df2["tau_to_zero"], label="tau_m -> 0")
plt.xlabel("z")
plt.ylabel("proxy amplitude (arb.)")
plt.title("P2: BAO Phase Tweak Proxy (Ablation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(pred/"plot_bao_tweak.png", dpi=160)
plt.close()

print("[pred] wrote plots:", pred/"plot_lensing_template.png", pred/"plot_bao_tweak.png")
