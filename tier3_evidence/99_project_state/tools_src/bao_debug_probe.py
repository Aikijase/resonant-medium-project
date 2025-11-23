#!/usr/bin/env python3
import numpy as np, pandas as pd, json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import joint_fit_resonant_rd as J

print("\n=== BAO DEBUG PROBE ===")

# --- Load BAO CSV ---
csv = "data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
cov = "data/desi_dr1_bao/bao_covariance_plus_lya.csv"
df = pd.read_csv(csv)
C  = pd.read_csv(cov, header=None).to_numpy(float)

cols = list(df.columns)
print(f"Columns: {cols}")
print(df.head(), "\n")

y = df["y_data"].to_numpy(float)
z = df["z"].to_numpy(float)
kind = df["kind"].astype(str).to_numpy()
print(f"len(y)={len(y)}, unique kind={sorted(set(kind))}, cov shape={C.shape}")

pack = {
    "y": y, "y_data": y, "z": z, "kind": kind,
    "C": C, "cov": C, "L": np.linalg.cholesky(C),
    "is_prec": False
}

# Split DM / DH if present
mask_dm = kind == "DM"
mask_dh = kind == "DH"
pack["y_DM"] = y[mask_dm]; pack["z_DM"] = z[mask_dm]
pack["y_DH"] = y[mask_dh]; pack["z_DH"] = z[mask_dh]

# --- Run make_bao_chi2_both ---
theta0 = np.array([2.0, 2.4, 1.4, 0.0])
try:
    chi2_bao = J.make_bao_chi2_both(
        bao_pack=pack, H0=70.0, Om0=0.3, rd0=147.1,
        fit_Om=False, fit_rd=False, nz_bg=800
    )(theta0)
    print("\nχ²_BAO =", chi2_bao)
except Exception as e:
    print("\nException during make_bao_chi2_both:", e)

print("=== END DEBUG ===")
