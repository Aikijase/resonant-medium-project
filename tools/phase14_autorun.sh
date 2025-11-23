#!/usr/bin/env bash
set -euo pipefail

OUTDIR="outputs/phase14"

# Narrow, medium, wide sweeps (tweak as desired)
python3 tools/phase14_lockmap.py \
  --omega2-min 2.70 --omega2-max 2.95 --omega2-step 0.01 \
  --Kphi-min 0.80  --Kphi-max 1.30  --Kphi-step 0.01 \
  --preset neuron --kv 0.10 --kx 0.20 --eps 0.06 \
  --noise 0.01 --steps 12000 --burn-in 300 \
  --outdir "$OUTDIR" --prefix p14_lockmap_narrow

python3 tools/phase14_lockmap.py \
  --omega2-min 2.50 --omega2-max 3.00 --omega2-step 0.02 \
  --Kphi-min 0.60  --Kphi-max 1.40  --Kphi-step 0.02 \
  --preset neuron --kv 0.10 --kx 0.20 --eps 0.06 \
  --noise 0.01 --steps 12000 --burn-in 300 \
  --outdir "$OUTDIR" --prefix p14_lockmap_medium
