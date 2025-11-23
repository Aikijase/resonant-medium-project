
#!/usr/bin/env bash
set -euo pipefail

# Example run script for Phase 2
mkdir -p data/bec outputs

# 1) Digitize (supply your own image path)
# python3 digitize_bec_ridge.py --image path/to/steinhauer_fig.png --out data/bec/steinhauer2016_ridgeA.csv --omega0 1.0 --label "BEC ridge A"

# 2) Overlay and fit universal law across systems
python3 collapse_compare_all.py   --inputs data/bec/steinhauer2016_ridgeA.csv data/abh_plate_norm.csv data/dusty_plasma_norm.csv   --out-prefix outputs/universal_collapse

# 3) Cosmic anchor (optional, supply rho_BH(z) CSV)
# python3 fit_cosmic_anchor.py --bh-density data/rho_BH_z.csv --out outputs/cosmic_anchor_fit
