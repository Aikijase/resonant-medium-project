#!/usr/bin/env bash
set -euo pipefail
echo "== Phase-5: Stability & Stress Test =="

mkdir -p outputs/phase5 plots/phase5

python3 tools/phase5_sweep_stability.py
python3 tools/phase5_damping_ode.py
python3 tools/phase5_summary.py

echo "Artifacts:"
ls -l outputs/phase5/stability_grid.json || true
ls -l outputs/phase5/stability_grid.csv  || true
ls -l outputs/phase5/damping_runs.json   || true
ls -l outputs/phase5/phase5_summary.md   || true
ls -l plots/phase5/phase5_stability_map.png || true
ls -l plots/phase5/phase5_damping_evolution.png || true
