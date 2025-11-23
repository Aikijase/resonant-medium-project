#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt, pathlib

OUTDIR = pathlib.Path("outputs/bao_limits_profile")
A = pd.read_csv(OUTDIR/"bao_limits_profile_analytic.csv").sort_values("f")
P = pd.read_csv(OUTDIR/"bao_limits_profile_profile.csv").sort_values("f")

# Merge and compute differences
df = A[["f","A_best","A95_up","delta_chi2"]].rename(
    columns={"A95_up":"A95_analytic","A_best":"A_best_free","delta_chi2":"dchi2"}
).merge(
    P[["f","A95_up"]].rename(columns={"A95_up":"A95_profile"}), on="f", how="inner"
)
df["A95_diff"]  = df["A95_profile"] - df["A95_analytic"]
df["A95_rel%"]  = 100.0 * df["A95_diff"] / df["A95_analytic"]

print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

# Plot A_best and both A95 curves
plt.figure()
plt.plot(df["f"], df["A_best_free"], marker="o", label="A_best (free)")
plt.plot(df["f"], df["A95_analytic"], marker="o", label="A95 (analytic)")
plt.plot(df["f"], df["A95_profile"], marker="o", label="A95 (profile)")
plt.xlabel("f")
plt.ylabel("Amplitude A")
plt.title("BAO amplitude best-fit and 95% upper limits vs f")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR/"A_best_A95_compare.png", dpi=160)
print(f"\nWrote: {OUTDIR/'A_best_A95_compare.png'}")
