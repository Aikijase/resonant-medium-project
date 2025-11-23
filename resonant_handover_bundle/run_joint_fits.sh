#!/usr/bin/env bash
set -euo pipefail

PROJ="${PROJ:-$HOME/resonant-medium-project}"
BAO_CSV="$PROJ/data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV="$PROJ/data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CAT="$PROJ/data/pantheon_plus/Pantheon+SH0ES.csv"
SN_COV_TXT="$PROJ/data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.cov"
SN_COV_FLOOR="$PROJ/data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.fixed_spd_floored.csv"
SN_DIR="$PROJ/data/sn"
SN_VEC="$SN_DIR/sn_pantheonplus_mb_MATCHED.csv"

OUT_LCDM="$PROJ/outputs/joint_lcdm_Pplus_FULLCOV_mb.json"
OUT_RESN="$PROJ/outputs/joint_resonant_rd_fixed_Pplus_FULLCOV_mb.json"

mkdir -p "$SN_DIR" "$PROJ/outputs"

python3 "/mnt/data/handover_bundle/tools/make_floored_cov.py" --in "$SN_COV_TXT" --out "$SN_COV_FLOOR" --floor-frac 1e-5
python3 "/mnt/data/handover_bundle/tools/sn_normalize.py"    --in "$SN_CAT"     --out "$SN_VEC"      --use-mb-corr --zcol zHD

stdbuf -oL -eL python3 -u "$HOME/sn_bao_joint_alpha_clean.py"   --bao-csv "$BAO_CSV" --bao-cov "$BAO_COV"   --sn-csv  "$SN_VEC"  --sn-cov  "$SN_COV_FLOOR"   --use-alpha-basis --fid-H0 70 --fid-Om 0.3 --fid-rd 147.1   --A-bounds 0,0 --f-bounds 1,1 --phi-bounds 0,0 --gamma-bounds 0,0   --rd-bounds 147.1,147.1   --fit A f phi gamma rd   --out "$OUT_LCDM"

stdbuf -oL -eL python3 -u "$HOME/sn_bao_joint_alpha_clean.py"   --bao-csv "$BAO_CSV" --bao-cov "$BAO_COV"   --sn-csv  "$SN_VEC"  --sn-cov  "$SN_COV_FLOOR"   --use-alpha-basis --fid-H0 70 --fid-Om 0.3 --fid-rd 147.1   --rd-bounds 147.1,147.1 --prior-A-sigma 0.4   --f-bounds 0.3,6.0 --gamma-bounds 0,1.5   --restarts 8   --fit A f phi gamma rd   --out "$OUT_RESN"

python3 "/mnt/data/handover_bundle/tools/compare_ic.py" --lcdm "$OUT_LCDM" --resn "$OUT_RESN"
