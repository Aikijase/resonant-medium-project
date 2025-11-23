#!/usr/bin/env bash
set -euo pipefail

# === CONFIG YOU MAY NEED TO EDIT ===
PY=${PY:-python3}
SCRIPT=${SCRIPT:-$HOME/sn_bao_joint_alpha_clean.py}   # point to your runner
OUTDIR=${OUTDIR:-$HOME/resonant-medium-project/outputs}
mkdir -p "$OUTDIR"

# Data (adjust paths as needed)
BAO_CSV=${BAO_CSV:-$HOME/resonant-medium-project/data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv}
BAO_COV=${BAO_COV:-$HOME/resonant-medium-project/data/desi_dr1_bao/bao_covariance_plus_lya.csv}
SN_CSV=${SN_CSV:-$HOME/resonant-medium-project/data/pantheon_plus/sn_MATCHED_mu.csv}
SN_COV=${SN_COV:-$HOME/resonant-medium-project/data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.spd.csv}

# Fiducials
FID_H0=${FID_H0:-70.0}
FID_OM=${FID_OM:-0.3}
FID_RD=${FID_RD:-147.1}

# --- Flag names (edit these to match your script) ---
# Curvature toggle:
#   Set CURV_OFF=""  and CURV_ON="--free-Ok"   (or whatever your script expects, e.g., --curvature, --ok-free, etc.)
CURV_OFF=""
CURV_ON="--free-Ok"

# r_d handling:
#   For fixed r_d, many pipelines simply fix via --rd-bounds <min,max> equal to fiducial.
#   For a prior, set --rd-prior-mean, --rd-prior-sigma if supported; otherwise just widen bounds.
RD_FIXED=(--rd-bounds "${FID_RD},${FID_RD}")
RD_PRIOR=(--rd-prior-mean "${FID_RD}" --rd-prior-sigma "2.5")  # adjust to your available flags

# Resonant model toggle:
#   Set RESN_OFF="" and RESN_ON="--use-alpha-basis" (or the flag that activates your resonance model).
RESN_OFF=""
RESN_ON="--use-alpha-basis"

# Common extras (set as needed)
EXTRAS=(--fid-H0 "${FID_H0}" --fid-Om "${FID_OM}" --fid-rd "${FID_RD}")

run_one () {
  local model_tag="$1"      # LCDM or RESN
  local curv_tag="$2"       # flat or curv
  local rd_tag="$3"         # fixed or prior

  local CURV_FLAGS=()
  if [[ "$curv_tag" == "curv" ]]; then
    CURV_FLAGS=(${CURV_ON})
  fi

  local RD_FLAGS=()
  if [[ "$rd_tag" == "fixed" ]]; then
    RD_FLAGS=("${RD_FIXED[@]}")
  else
    RD_FLAGS=("${RD_PRIOR[@]}")
  fi

  local RESN_FLAGS=()
  if [[ "$model_tag" == "RESN" ]]; then
    RESN_FLAGS=(${RESN_ON})
  fi

  local OUT="${OUTDIR}/poc_${model_tag}_${curv_tag}_${rd_tag}.json"
  echo ">>> Running ${model_tag} ${curv_tag} ${rd_tag} -> ${OUT}"

  # You may need to add --out "${OUT}" if your script writes JSON. Otherwise,
  # you can pipe stdout to a file and add that path into the JSON as 'outfile' after.
  CMD=( "${PY}" -u "${SCRIPT}"
        --bao-csv "${BAO_CSV}" --bao-cov "${BAO_COV}"
        --sn-csv "${SN_CSV}"   --sn-cov "${SN_COV}"
        "${EXTRAS[@]}"
        "${CURV_FLAGS[@]}"
        "${RD_FLAGS[@]}"
        "${RESN_FLAGS[@]}"
        --out "${OUT}" )

  echo "${CMD[@]}"
  "${CMD[@]}"
}

# === MATRIX ===
# 1) LCDM flat, r_d fixed
run_one LCDM flat fixed
# 2) LCDM curv, r_d fixed
run_one LCDM curv fixed
# 3) LCDM flat, r_d prior
run_one LCDM flat prior
# 4) LCDM curv, r_d prior
run_one LCDM curv prior

# 5) RESN flat, r_d fixed
run_one RESN flat fixed
# 6) RESN curv, r_d fixed
run_one RESN curv fixed
# 7) RESN flat, r_d prior
run_one RESN flat prior
# 8) RESN curv, r_d prior
run_one RESN curv prior

echo "All runs finished. Now compute AIC/BIC with calc_aic_bic.py"
echo "Example:"
echo "python3 calc_aic_bic.py --json-glob "$OUTDIR/poc_*.json" --log poc_master_log.csv --N_SN 1701 --N_BAO 26 --N_prior 1 --baseline "LCDM:curv=0:rd=fixed" "LCDM:curv=1:rd=fixed" "LCDM:curv=0:rd=prior" "LCDM:curv=1:rd=prior""
