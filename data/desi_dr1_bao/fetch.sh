set -euo pipefail
base='https://raw.githubusercontent.com/CobayaSampler/bao_data/master'

# BGS (DV only)
curl -fsSLo data/desi_dr1_bao/BGS_mean.txt "$base/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt"
curl -fsSLo data/desi_dr1_bao/BGS_cov.txt  "$base/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_cov.txt"

# LRG 0.4–0.6
curl -fsSLo data/desi_dr1_bao/LRG1_mean.txt "$base/desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt"
curl -fsSLo data/desi_dr1_bao/LRG1_cov.txt  "$base/desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt"

# LRG 0.6–0.8
curl -fsSLo data/desi_dr1_bao/LRG2_mean.txt "$base/desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt"
curl -fsSLo data/desi_dr1_bao/LRG2_cov.txt  "$base/desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt"

# LRG+ELG 0.8–1.1 (highest-precision bin)
curl -fsSLo data/desi_dr1_bao/LRGELG_mean.txt "$base/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt"
curl -fsSLo data/desi_dr1_bao/LRGELG_cov.txt  "$base/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt"

# ELG 1.1–1.6
curl -fsSLo data/desi_dr1_bao/ELG_mean.txt "$base/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt"
curl -fsSLo data/desi_dr1_bao/ELG_cov.txt  "$base/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt"

# QSO 0.8–2.1 (DV only)
curl -fsSLo data/desi_dr1_bao/QSO_mean.txt "$base/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt"
curl -fsSLo data/desi_dr1_bao/QSO_cov.txt  "$base/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt"

# (Optional) Lyα z=2.33
curl -fsSLo data/desi_dr1_bao/LYA_mean.txt "$base/desi_2024_gaussian_bao_Lya_GCcomb_mean.txt"
curl -fsSLo data/desi_dr1_bao/LYA_cov.txt  "$base/desi_2024_gaussian_bao_Lya_GCcomb_cov.txt"
echo "Fetched DESI DR1 BAO Gaussian 'mean' and 'cov' files into data/desi_dr1_bao/"
