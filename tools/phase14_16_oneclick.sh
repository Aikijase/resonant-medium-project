#!/usr/bin/env bash
set -Eeuo pipefail

SHELL_CSV="${1:?usage: $0 <shell_csv>}"
OUT14="outputs/phase14"
OUT15="outputs/phase15"
OUT16="outputs/phase16"
LOGDIR="outputs/logs"
PREFIX="p14_edge"

mkdir -p "$OUT14" "$OUT15" "$OUT16" "$LOGDIR"

ts() { date +%Y%m%d-%H%M%S; }
log() { echo "[oneclick] $*"; }

log "Phase-14 widths & areas..."
P14W_LOG="$LOGDIR/phase14_widths_$(ts).log"
python3 tools/phase14_contour_width.py \
  --shell-csv "$SHELL_CSV" \
  --outdir "$OUT14" --prefix "$PREFIX" \
  --levels 0.90,0.92,0.94,0.96,0.98 \
  --low 0.90 --high 0.98 \
  --nw 600 --nk 600 --wbins 900 | tee "$P14W_LOG"

# Batch + normalize + report
log "Phase-14 batch summary..."
P14B_LOG="$LOGDIR/phase14_batch_$(ts).log"
python3 tools/phase14_batch_widths.py \
  --shell-csv "$SHELL_CSV" \
  --outdir "$OUT14" --prefix "$PREFIX" \
  --levels 0.90,0.92,0.94,0.96,0.98 \
  --pairs 0.90:0.98,0.92:0.98,0.94:0.98 | tee "$P14B_LOG"

LATEST_JSON="$(ls -1t "$OUT14"/${PREFIX}_batch_summary_*.json | head -n1)"
log "Normalize areas..."
python3 tools/phase14_normalize_areas.py \
  --shell-csv "$SHELL_CSV" \
  --summary-json "$LATEST_JSON" \
  --outdir "$OUT14" --prefix "$PREFIX" | tee "$LOGDIR/phase14_norm_$(ts).log"

python3 tools/phase14_make_report_card.py \
  --summary-json "$LATEST_JSON" \
  --outdir "$OUT14" --prefix "$PREFIX" | tee "$LOGDIR/phase14_rc_$(ts).log"

# Centerline (capture stdout to parse curvature cleanly)
log "Phase-14 centerline..."
P14C_LOG="$LOGDIR/phase14_centerline_$(ts).log"
python3 tools/phase14_centerline.py \
  --contour-low  "$OUT14/${PREFIX}_contour_si0.900.csv" \
  --contour-high "$OUT14/${PREFIX}_contour_si0.980.csv" \
  --low 0.90 --high 0.98 \
  --wbins 900 --outdir "$OUT14" --prefix "$PREFIX" | tee "$P14C_LOG"

# Extract centerline curvature from the captured stdout
C_CENTER="$(grep -Eo 'curvature_c \(quadratic coefficient\): *[0-9.eE+-]+' "$P14C_LOG" | awk '{print $NF}')"
CENTER_CSV="$OUT14/${PREFIX}_centerline_si0.900_to_si0.980.csv"

# Ridge vs centerline (capture ridge fit coefficients)
log "Phase-15 ridge vs centerline..."
P15R_LOG="$LOGDIR/phase15_ridge_$(ts).log"
python3 tools/phase15_ridge_from_grid.py \
  --shell-csv "$SHELL_CSV" \
  --centerline-csv "$CENTER_CSV" \
  --outdir "$OUT15" --prefix p15_cmp --nw 600 --nk 600 | tee "$P15R_LOG"

# --- robust parse of ridge_quadratic_fit line(s) ---
extract_coef() {
  # $1 is key: a | b | c | R2
  awk -v key="$1" '
  /ridge_quadratic_fit:/{
    for(i=1;i<=NF;i++){
      if ($i ~ ("^"key"=")) { val=$i; sub(key"=","",val); print val }
    }
  }' "$P15R_LOG" | tail -n1
}

A="$(extract_coef a || echo 0)"
B="$(extract_coef b || echo 0)"
C="$(extract_coef c || echo 0)"
R2="$(extract_coef R2 || echo 0)"

python3 tools/phase15_make_report_card.py \
  --ridge-csv "$OUT15/p15_cmp_ridge.csv" \
  --centerline-csv "$CENTER_CSV" \
  --ridge-png "$OUT15/p15_cmp_ridge.png" \
  --cmp-png "$OUT15/p15_cmp_ridge_vs_centerline.png" \
  --shell-csv "$SHELL_CSV" \
  --outdir "$OUT15" --prefix p15_cmp \
  --c-center "${C_CENTER:-0}" \
  --a "${A:-0}" --b "${B:-0}" --c "${C:-0}" --R2 "${R2:-0}" | tee "$LOGDIR/phase15_rc_$(ts).log"

# Phase-16 compare (single shell) + report
log "Phase-16 compare + report..."
python3 tools/phase16_compare_shells.py \
  --shells "$SHELL_CSV" \
  --outdir "$OUT16" --prefix p16_clean | tee "$LOGDIR/phase16_cmp_$(ts).log"

python3 tools/phase16_make_report_card.py \
  --summary-json "$OUT16/p16_clean_summary.json" \
  --outdir "$OUT16" --prefix p16_clean | tee "$LOGDIR/phase16_rc_$(ts).log"

# Final SITREP
echo
echo "=== ONECLICK SITREP ==="
echo "Centerline curvature c_center = ${C_CENTER:-NA}"
echo "Ridge curvature c_ridge = ${C:-NA}  (R2=${R2:-NA})"
echo "Normalized area images: $OUT16/p16_clean_A_mask_norm.png"
echo "Ridge vs centerline:   $OUT15/p15_cmp_ridge_vs_centerline.png"
echo "Logs in:               $LOGDIR/"
echo "========================"
