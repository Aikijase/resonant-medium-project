#!/usr/bin/env bash
set -euo pipefail
MAIN="paper-jcap/main.tex"
TMP="${MAIN}.tmp"

# Wrap the table's middle numeric cell (between the first & and next &)
# for any row that starts with our parameter names.
awk '
function wrap_math(s) {
  gsub(/^[ \t]+|[ \t]+$/, "", s)
  if (s ~ /^\$/ && s ~ /\$$/) return s
  return "$" s "$"
}
{
  line = $0
  if (line ~ /\$\\tauMem\$/ || line ~ /\$\\kappaMem\$/ || line ~ /\$\\wzero\$/ || line ~ /\$\\Qfact\$/ || line ~ /\$C_\\mathrm\{cal\}\$/) {
    # find first &
    amp1 = index(line, "&")
    if (amp1 > 0) {
      # find second & after amp1
      rest = substr(line, amp1+1)
      amp2rel = index(rest, "&")
      if (amp2rel > 0) {
        before   = substr(line, 1, amp1)           # up to and including first &
        middle   = substr(rest, 1, amp2rel-1)      # between &
        after    = substr(rest, amp2rel)           # from second & onwards (includes &)
        # ensure middle is math-wrapped
        gsub(/\r$/, "", middle)                    # strip CR if any
        if (middle !~ /\$.*\$/) {
          middle = " " wrap_math(middle) " "
        }
        line = before middle after
      }
    }
  }
  print line
}' "$MAIN" > "$TMP" && mv "$TMP" "$MAIN"

echo "✔ Wrapped table numeric cells in math mode: $MAIN"
