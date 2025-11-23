#!/usr/bin/env bash
set -euo pipefail
echo "== Phase-4: multi-bin g×κ =="

mkdir -p outputs/phase4 plots

python3 tools/phase4_gk_multibin_score.py \
  --bins data/isw/gk_bins.csv \
  --scale-file outputs/phase2/growth_scale.json \
  --out outputs/phase4/gk_multi.json

python3 tools/phase4_gk_multibin_table.py
python3 tools/phase4_gk_multibin_plot.py

echo "Artifacts:"
ls -l outputs/phase4/gk_multi.json || true
ls -l outputs/phase4/gk_multi_table.md || true
ls -l plots/gk_multi_residuals.png || true
