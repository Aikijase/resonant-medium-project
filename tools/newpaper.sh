#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  newpaper.sh <slug> ["Full Paper Title"] [--import /path/overleaf.zip]

Examples:
  newpaper.sh framework "Resonant Medium: A Framework"
  newpaper.sh cosmic-recycling "Cosmic Recycling via Resonant Media" --import ~/Downloads/cosmic.zip

Notes:
  - Creates papers/<slug>/ structure (via tools/setup_paper.sh)
  - Optionally imports an Overleaf zip into manuscript (via tools/overleaf_sync.sh import)
  - Always exports a clean Overleaf-ready zip to outputs/
EOF
  exit 1
}

[[ $# -lt 1 ]] && usage
SLUG="$1"; shift || true
TITLE="${1:-$SLUG}"
# If the second arg starts with '--', treat as no-title case
if [[ "${TITLE}" == --* ]]; then
  TITLE="$SLUG"
else
  shift || true
fi

IMPORT_ZIP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --import) shift; IMPORT_ZIP="${1:-}";;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
  shift || true
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1) Ensure prereqs exist
[[ -x "$ROOT/tools/setup_paper.sh" ]] || { echo "Missing or not executable: tools/setup_paper.sh"; exit 1; }
[[ -x "$ROOT/tools/overleaf_sync.sh" ]] || { echo "Missing or not executable: tools/overleaf_sync.sh"; exit 1; }

# 2) Create the paper workspace
bash "$ROOT/tools/setup_paper.sh" "$SLUG" "$TITLE"

# 3) Optional: import Overleaf zip
if [[ -n "$IMPORT_ZIP" ]]; then
  bash "$ROOT/tools/overleaf_sync.sh" import "$SLUG" "$IMPORT_ZIP"
fi

# 4) Always make an initial Overleaf-ready upload zip
bash "$ROOT/tools/overleaf_sync.sh" export "$SLUG" "$ROOT/outputs"

echo "✅ Done. Paper: papers/${SLUG}"
