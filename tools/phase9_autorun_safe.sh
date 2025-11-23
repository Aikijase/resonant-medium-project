#!/usr/bin/env bash
# Phase 9 autorun (no-scipy) — safe mode with logs
mkdir -p logs outputs/phase9

run() {
  local title="$1"; shift
  echo "===== $title ====="
  echo "\$ $*" | tee -a "logs/phase9_${title// /_}.log"
  { "$@" 2>&1 | tee -a "logs/phase9_${title// /_}.log"; } || \
     echo "[warn] step failed: $title" | tee -a "logs/phase9_${title// /_}.log"
  echo
}

run "physical"        env PYTHONPATH=. MPLBACKEND=Agg python3 tools/phase9_unified_operator_sim_noscipy.py --config configs/phase9_physical.yaml --out-prefix outputs/phase9/physical_run
run "psychological"   env PYTHONPATH=. MPLBACKEND=Agg python3 tools/phase9_unified_operator_sim_noscipy.py --config configs/phase9_psychological.yaml --out-prefix outputs/phase9/psychological_run_noscipy
run "learning"        env PYTHONPATH=. MPLBACKEND=Agg python3 tools/phase9_unified_operator_sim_noscipy.py --config configs/phase9_learning.yaml --out-prefix outputs/phase9/learning_run_noscipy
run "metrics"         env PYTHONPATH=. python3 tools/phase9_compare_metrics.py outputs/phase9/physical_run.csv outputs/phase9/psychological_run_noscipy.csv outputs/phase9/learning_run_noscipy.csv
run "overlay_plots"   env PYTHONPATH=. python3 tools/phase9_plot_overlay.py outputs/phase9/physical_run.csv outputs/phase9/psychological_run_noscipy.csv outputs/phase9/learning_run_noscipy.csv
run "report_card"     env PYTHONPATH=. python3 tools/phase9_report_card.py outputs/phase9/physical_run.csv outputs/phase9/psychological_run_noscipy.csv outputs/phase9/learning_run_noscipy.csv

echo "[phase9] Safe autorun finished. See outputs/phase9 and logs/."
