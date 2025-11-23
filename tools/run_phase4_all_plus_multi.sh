#!/usr/bin/env bash
set -euo pipefail
echo "== Phase-4: all-up =="

mkdir -p outputs/phase4 plots

# Single-catalog flow
python3 tools/phase4_gk_score.py \
  --catalogs data/isw/gk_catalogs.csv \
  --scale-file outputs/phase2/growth_scale.json \
  --out outputs/phase4/gk_score.json
python3 tools/phase4_gk_plot.py
python3 tools/phase4_summary.py
python3 tools/phase4_report_patch.py
python3 tools/phase4_table_export.py

# Multi-bin flow
python3 tools/phase4_gk_multibin_score.py \
  --bins data/isw/gk_bins.csv \
  --scale-file outputs/phase2/growth_scale.json \
  --out outputs/phase4/gk_multi.json
python3 tools/phase4_gk_multibin_table.py
python3 tools/phase4_gk_multibin_plot.py
python3 tools/phase4_multibin_summary.py
python3 tools/phase4_multibin_report_patch.py

echo "Artifacts:"
ls -l outputs/phase4/gk_score.json        || true
ls -l outputs/phase4/gk_table.md          || true
ls -l plots/gk_residuals.png              || true
ls -l outputs/phase4/gk_multi.json        || true
ls -l outputs/phase4/gk_multi_table.md    || true
ls -l plots/gk_multi_residuals.png        || true
