#!/usr/bin/env bash
set -euo pipefail
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$(dirname "$THIS_DIR")/.."

cd "$ROOT"
python3 tools/phase2_smallscale/run.py --config tools/phase2_smallscale/config_smallscale.yaml
echo "Done. See outputs/phase2_smallscale/ for figures and JSON summaries."
