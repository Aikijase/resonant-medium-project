# tools/run_phase4_gk.sh
#!/usr/bin/env bash
set -euo pipefail
echo "== Phase-4: g×κ shape-only check =="

# Optional: provide a model scale via JSON:
#   echo '{"A_gk_scale": 0.97}' > outputs/phase2/growth_scale.json
# If absent, scorer defaults to 1.0 (neutral).

python3 tools/phase4_gk_score.py \
  --catalogs data/isw/gk_catalogs.csv \
  --scale-file outputs/phase2/growth_scale.json \
  --out outputs/phase4/gk_score.json

echo "Artifacts:"
ls -l outputs/phase4/gk_score.json || true
