#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   tools/phase8_spectral/view_outputs.sh <out_stem>
# Example:
#   tools/phase8_spectral/view_outputs.sh outputs/phase8/ocean_memory_tau0p30_kap-0p02_hiRes

stem="${1:-outputs/phase8/ocean_memory_tau0p30_kap-0p02_hiRes}"

summary="${stem}_summary.json"
series="${stem}_residual_series.csv"
plot="${stem}_spectrum.png"

echo "== Stem =="
echo "$stem"
echo

echo "== Files =="
printf "%-48s : %s\n" "$summary" "$( [ -f "$summary" ] && echo OK || echo MISSING )"
printf "%-48s : %s\n" "$series"  "$( [ -f "$series"  ] && echo OK || echo MISSING )"
printf "%-48s : %s\n" "$plot"    "$( [ -f "$plot"    ] && echo OK || echo MISSING )"
echo

if [[ -f "$summary" ]]; then
  echo "== Summary (pretty) =="
  python3 - <<'PY'
import json, os
p = os.environ["SUMMARY_PATH"]
with open(p) as f:
    J = json.load(f)
for k in sorted(J):
    print(f"{k:22s}: {J[k]}")
PY
else
  echo "(summary not found)"
fi
echo

echo "== Residual series (head) =="
if [[ -f "$series" ]]; then
  head -n 10 "$series" || true
else
  echo "(missing)"
fi
echo

echo "== Open plot =="
if [[ -f "$plot" ]]; then
  xdg-open "$plot" >/dev/null 2>&1 || echo "(couldn't auto-open image)"
else
  echo "(missing)"
fi
