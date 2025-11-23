#!/usr/bin/env bash
set -euo pipefail
echo "== Phase-4: real-data pipeline =="

# 1) Predict A_gk_scale from your Phase-2 growth outputs and current gk_bins
python3 tools/phase4_compute_agk_from_fs8.py

# 2) Run the full Phase-4 stack (single + multi-bin, summaries, tables, plots, report)
bash tools/run_phase4_all_plus_multi.sh

# 3) Pack artifacts
python3 tools/phase4_pack_artifacts.py
