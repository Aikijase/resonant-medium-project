#!/usr/bin/env bash
set -euo pipefail
echo "== Phase-4: orchestrator =="

mkdir -p outputs/phase4 plots

# g×κ scorer
python3 tools/phase4_gk_score.py \
  --catalogs data/isw/gk_catalogs.csv \
  --scale-file outputs/phase2/growth_scale.json \
  --out outputs/phase4/gk_score.json

# Plot
python3 tools/phase4_gk_plot.py

# Summary
python3 tools/phase4_summary.py

# Report patch
python3 tools/phase4_report_patch.py

# Table export
python3 tools/phase4_table_export.py

echo "Artifacts:"
ls -l outputs/phase4/gk_score.json || true
ls -l outputs/phase4/gk_table.md   || true
ls -l plots/gk_residuals.png       || true

