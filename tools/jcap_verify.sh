#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT/paper-jcap"

echo "Checking figure paths..."
grep -Eo '\\includegraphics\[.*\]\{[^}]+\}' main.tex | sed -E 's/.*\{([^}]+)\}.*/\1/' | while read -r p; do
  [[ -f "$p" ]] || echo "MISSING: $p"
done

echo "Fixing unicode and typography..."
../tools/jcap_fix_unicode.sh

echo "Filling posterior values..."
python3 ../tools/jcap_fill_posteriors.py --fit ../outputs/phase20/p20_fit.json --tex main.tex --out main.tex

echo "Building PDF..."
latexmk -pdf main.tex

echo "Done. Output saved to paper-jcap/main.pdf"
