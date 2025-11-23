
#!/usr/bin/env bash
# ============================
# File: tools/thrace_bootstrap.sh
# ----------------------------
# Bootstrap a local venv (PEP 668-friendly), install deps, and run tests.
# Usage (from project root):
#   bash tools/thrace_bootstrap.sh
set -euo pipefail

# Ensure python3-venv exists (Debian/Ubuntu/Chromebook). Suggest, don't sudo automatically.
if ! python3 -c "import venv" >/dev/null 2>&1; then
  echo "[thrace] python3-venv not found. On Debian/Ubuntu try: sudo apt-get install -y python3-venv python3-dev" >&2
  exit 1
fi

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[thrace] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --upgrade numpy scipy pandas pyyaml

# Make sure package path works
python - <<'PY'
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import tools.thrace_recycling_model as _
print('[thrace] import check OK from', root)
PY

# Run tests
PYTHONPATH=. python -m tools.test_thrace

echo "[thrace] Bootstrap complete. Next:"
echo "  source $VENV_DIR/bin/activate"
echo "  PYTHONPATH=. python tools/thrace_orchestrator.py --config configs/thrace_default.yaml"
