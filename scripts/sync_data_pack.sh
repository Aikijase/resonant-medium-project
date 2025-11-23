#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 /path/to/recycling_data_consolidation_pack_v*.zip [--dry-run]"
  echo "Example: $0 \"/mnt/chromeos/MyFiles/Downloads/recycling_data_consolidation_pack_v1_7.zip\" --dry-run"
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi

ZIP_PATH="$1"; shift || true
DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then DRY="--dry-run"; shift || true; fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "ERROR: ZIP not found at: $ZIP_PATH"
  exit 2
fi

TMP_DIR="$(mktemp -d -t data_pack_sync_XXXXXX)"
unzip -q "$ZIP_PATH" -d "$TMP_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR=".backup_data_pack_$TS"
mkdir -p "$BACKUP_DIR"

RSYNC_COMMON=( -a --human-readable --info=stats2,progress2
  --backup --backup-dir="$BACKUP_DIR"
  --exclude ".git/" --exclude ".github/" --exclude ".venv/" --exclude "node_modules/"
)

if [[ -n "$DRY" ]]; then
  echo ">>> DRY RUN (no files will be changed)"
  rsync -n --itemize-changes "${RSYNC_COMMON[@]}" "$TMP_DIR"/ ./
else
  rsync    --itemize-changes "${RSYNC_COMMON[@]}" "$TMP_DIR"/ ./
  if compgen -G "tools/data/*.py" > /dev/null; then
    chmod +x tools/data/*.py || true
  fi
fi

echo
echo "=== Sync complete ==="
if [[ -n "$DRY" ]]; then
  echo "No files changed (dry run). Remove --dry-run to apply."
else
  echo "Backups of any overwritten files are in: $BACKUP_DIR"
  echo "Tip: to restore: cp -a \"$BACKUP_DIR/path/to/file\" path/to/file"
fi
