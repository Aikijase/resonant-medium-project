#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/paper-jcap"

fix() { local c="$1" r="$2"; LC_ALL=C grep -RIl --null "$c" "$DIR" | while IFS= read -r -d '' f; do sed -i "s/$c/$r/g" "$f"; done || true; }

# Common LaTeX-unsafe unicode
fix "≈" "\\\\approx{}"
fix "±" "\\\\pm{}"
fix "–" "--"
fix "—" "---"
fix "°" "\\\\degree{}"

# Smart quotes/apostrophes
fix "“" "``"
fix "”" "''"
fix "’" "'"

echo "Unicode/typography sweep complete in $DIR"
