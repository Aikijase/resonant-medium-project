#!/usr/bin/env bash
set -euo pipefail
python3 tools/phase2_smallscale_sweep.py
echo "Now try a dwarf-count estimate:"
echo "python3 tools/phase2_dwarfcount.py --tau 0.05 --omega 0.144 --Mhost 1e12 --Mmin 1e9 --Mmax 1e10 --Rvir 250"
