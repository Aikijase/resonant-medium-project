#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${1:-$HOME/cosmic_resonance}"
echo "Creating project at: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"/{data/{planck2018,pantheon_plus,desi_dr1_bao,des_y3,kids_1000,hsc_y3,growth_fsigma8},code/{loaders,models},likelihoods,outputs/{chains,plots,metrics}}
echo "Done."
echo "Tip: move downloaded files into the matching data/* folders."
