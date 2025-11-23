#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}

echo "== Phase-4: CMB lensing cross-check =="
mkdir -p outputs/phase4 plots

$PY tools/score_cmb_lensing.py
$PY plots/plot_cmb_lensing_amp.py
$PY tools/finalize_phase4.py

echo "Artifacts:"
ls -l outputs/phase4/lensing_score.json plots/cmb_lensing_amp.png
