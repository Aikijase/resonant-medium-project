#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SRC="$ROOT/paper-jcap"
OUT="$ROOT/paper-jcap-submit"
rm -rf "$OUT"
mkdir -p "$OUT/figs"

# essentials
cp "$SRC/main.tex" "$OUT/"
cp "$SRC/refs.bib" "$OUT/" 2>/dev/null || true
cp "$SRC/main.pdf" "$OUT/" 2>/dev/null || true
cp "$SRC/latexmkrc" "$OUT/" 2>/dev/null || true

# figures (vector PDFs preferred)
cp "$SRC/figs/"*.pdf "$OUT/figs/" 2>/dev/null || true

# do NOT include local jcappub.sty; Overleaf provides the real class
echo "Ready to upload: $OUT"
