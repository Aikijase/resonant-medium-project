#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmd="${1:-}"; shift || true

timestamp() { date +%Y%m%d-%H%M%S; }

case "$cmd" in
  import)
    slug="$1"; zip="$2"; shift 2
    mkdir -p "$root/papers/$slug/manuscript/overleaf_src"
    dest="$root/papers/$slug/manuscript/overleaf_src/$(timestamp)"
    unzip -q "$zip" -d "$dest"
    rsync -av --include='*.tex' --include='*.bib' --exclude='*' "$dest"/ "$root/papers/$slug/manuscript/"
    echo "✅ Imported Overleaf zip to $dest"
    ;;
  export)
    slug="$1"; out="${2:-$root/outputs}"
    mkdir -p "$out"
    zip="$out/${slug}_overleaf_$(timestamp).zip"
    (cd "$root/papers/$slug/manuscript" && zip -qr "$zip" .)
    echo "✅ Exported manuscript zip to $zip"
    ;;
  *)
    echo "Usage: overleaf_sync.sh import <slug> <zip> | export <slug> [outdir]"
    exit 1;;
esac
