#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}

echo "== Phase-3: γ(z) evolution stress test =="
[[ -f configs/phase3.toml ]] || { echo "Missing configs/phase3.toml"; exit 1; }
mkdir -p outputs/phase3 plots

$PY phase3/compute_fs8_gammaevo.py --cfg configs/phase3.toml
$PY plots/plot_phase3_gammaevo_heatmap.py

echo "Artifacts:"
ls -l outputs/phase3/fs8_gammaevo_grid.csv outputs/phase3/fs8_gammaevo_best.json plots/phase3_gammaevo_heatmap.png
