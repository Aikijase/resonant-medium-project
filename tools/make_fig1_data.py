import numpy as np

# ---------- 1. Load Pantheon+ SN binned data ----------
sn_path = "data/pantheon_plus/sn_MATCHED_mu.csv"
sn = np.loadtxt(sn_path, delimiter=",", skiprows=1)

z_sn    = sn[:, 0]
mu_data = sn[:, 1]
mu_err  = sn[:, 2]

# ---------- 2. Load DESI DR1 BAO compressed data ----------
bao_path = "data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
bao = np.genfromtxt(bao_path, delimiter=",", names=True, dtype=None, encoding=None)

mask_DM   = (bao["kind"] == "DM")
z_bao     = bao["z"][mask_DM]
bao_data  = bao["y_data"][mask_DM]
bao_err   = bao["sigma"][mask_DM]

# ---------- 3. Save bundle for later plotting ----------
np.savez(
    "outputs/fig1_data.npz",
    z_sn=z_sn,
    mu_data=mu_data,
    mu_err=mu_err,
    z_bao=z_bao,
    bao_data=bao_data,
    bao_err=bao_err,
)

print("Wrote outputs/fig1_data.npz with SN + BAO data.")
