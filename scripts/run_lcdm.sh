#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs logs

# --- Edit this block if you want a different fitter/flags ---
CMD=(
  python3 joint_fit_resonant_anchors.py
  --bao-csv data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv
  --bao-cov data/desi_dr1_bao/bao_covariance_plus_lya.csv
  --sn-csv  data/pantheon_plus/sn_MATCHED_mu.csv
  --sn-cov  outputs/Pantheon+SH0ES_STAT+SYS.cal.cov
  --H0 70.0 --Om 0.3 --rd 147.1
  --A 0.0 --f 0.0 --phi 0.0 --gamma 0.0     # force ΛCDM-equivalent
)
# -----------------------------------------------------------

OUT_PREFIX=outputs/joint_LCDM_baseline
LOG=logs/lcdm_baseline.stdout
echo "[RUN]" "${CMD[@]}"
"${CMD[@]}" | tee "$LOG"
python3 tools/grab_last_json.py "$LOG" "${OUT_PREFIX}.json"
python3 tools/metrics.py "${OUT_PREFIX}.json"
