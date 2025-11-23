#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}

echo "== Phase-2 pipeline =="
echo "Using Python: $(command -v $PY)"

# 0) sanity: required files
req_files=("configs/resonant.toml" "growth.py" "data/growth/fs8_catalog.csv")
for f in "${req_files[@]}"; do
  [[ -f "$f" ]] || { echo "Missing required file: $f"; exit 1; }
done
mkdir -p outputs/phase2 plots data/isw tools

# 1) ΛCDM reference for fs8 (for ΔAIC/ΔBIC/WWI)
if [[ -f tools/make_lcdm_fs8_ref.py ]]; then
  echo "--> Generating ΛCDM fs8 reference"
  $PY tools/make_lcdm_fs8_ref.py
else
  echo "WARN: tools/make_lcdm_fs8_ref.py not found; fs8 deltas/WWI will be None"
fi

# 2) fs8 eval (resonant) + plots
echo "--> Running fs8 evaluation"
$PY run_growth_fs8.py --in configs/resonant.toml --out outputs/phase2/fs8_eval.json ${FS8_REF:+--ref "$FS8_REF"}
# If a reference exists from step 1, use it; otherwise skip deltas
if [[ -f outputs/phase2/fs8_eval_lcdm.json ]]; then
  $PY run_growth_fs8.py --in configs/resonant.toml --out outputs/phase2/fs8_eval.json --ref outputs/phase2/fs8_eval_lcdm.json
fi
echo "--> Making fs8 plots"
$PY plots/plot_fs8_overlay.py
if [[ -f plots/plot_fs8_residuals.py ]]; then
  $PY plots/plot_fs8_residuals.py
fi

# 3) ISW proxy + (optional) scoring
  echo "--> ISW proxy bandpowers"
  $PY run_isw_xcorr.py --in configs/resonant.toml --out outputs/phase2/isw_eval.json
if [[ -f data/isw/bandpowers.csv && -f tools/score_isw_from_csv.py ]]; then
  echo "--> Scoring ISW against CSV data"
  $PY tools/score_isw_from_csv.py || true
  echo "--> ISW amplitude scoring (WISE + RACS)"
  $PY tools/score_isw_from_csv.py || true
if [[ -f plots/plot_isw_amplitude.py ]]; then
  echo "--> ISW amplitude figure"
  $PY plots/plot_isw_amplitude.py || true
fi
fi

# 4) Stability sweep + heatmap
if [[ -f tools/make_stability_map_stream.py ]]; then
  echo "--> Stability sweep (streaming)"
  $PY tools/make_stability_map_stream.py
else
  echo "--> Stability sweep"
  $PY tools/make_stability_map.py
fi
echo "--> Plotting stability heatmap"
$PY plots/plot_stability_heatmap.py

# 5) Joint summary + report
if [[ -f tools/joint_guard_phase2.py ]]; then
  echo "--> Joint summary"
  $PY tools/joint_guard_phase2.py
fi
if [[ -f tools/make_phase2_report.py ]]; then
  echo "--> REPORT.md"
  $PY tools/make_phase2_report.py
fi
if [[ -f tools/print_phase2_summary.py ]]; then
  echo "--> Console summary"
  $PY tools/print_phase2_summary.py
fi
echo "--> Finalize & update REPORT"
python3 tools/finalize_phase2.py

echo "== Done =="
echo "Artifacts in outputs/phase2 and plots/"
python3 tools/run_phase4.sh
python3 tools/run_phase4_gk.sh
bash tools/run_phase4_all.sh
