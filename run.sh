#!/usr/bin/env bash
# ======================================================
# Resonant Medium Project — One-Step Run Script
# ======================================================
# Usage:
#   bash run.sh
#
# This script:
#   1. Activates the venv
#   2. Runs the universality collapse analysis (NRMSE metric)
#   3. Prints a summary
#   4. Optionally opens the result plots if desktop viewer available
# ======================================================

set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT"

# 1. Activate venv
if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
else
    echo "[WARN] No .venv found, creating one..."
    python3 -m venv .venv
    source "$ROOT/.venv/bin/activate"
    pip install -U pip
    pip install numpy pandas matplotlib scipy pyyaml bottleneck
fi

# 2. Run the collapse analysis
echo "------------------------------------------------------"
echo "Running universality collapse (metric = NRMSE)"
echo "------------------------------------------------------"
python3 collapse_from_registry.py --metric nrmse

# 3. Display summary
echo "------------------------------------------------------"
echo " Collapse Summary"
echo "------------------------------------------------------"
if [ -f "$ROOT/outputs/collapse_report.txt" ]; then
    cat "$ROOT/outputs/collapse_report.txt"
else
    echo "[WARN] No collapse_report.txt found."
fi

# 4. Auto-open plots (if viewer available)
if command -v xdg-open >/dev/null 2>&1; then
    echo "------------------------------------------------------"
    echo "Opening plots..."
    echo "------------------------------------------------------"
    xdg-open "$ROOT/outputs/collapse_eps_fit.png" >/dev/null 2>&1 &
    xdg-open "$ROOT/outputs/collapse_gamma.png" >/dev/null 2>&1 &
else
    echo "[INFO] Plots generated in outputs/: collapse_eps_fit.png and collapse_gamma.png"
fi

echo "------------------------------------------------------"
echo "Run complete ✅"
echo "------------------------------------------------------"
