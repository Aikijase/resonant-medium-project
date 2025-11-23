#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
MAIN="$ROOT/paper-jcap/main.tex"

# 1) First do the full rescue build (figs, stub, helpers, fill, compile)
"$ROOT/tools/jcap_build_rescue.sh" || true

# 2) Force-wrap ANY table middle cell that looks numeric or uses \pm, if not already in $...$
python3 - <<'PY'
import re, pathlib, sys, os
root = pathlib.Path(os.environ["ROOT"]) if "ROOT" in os.environ else pathlib.Path(".").resolve()
main = root / "paper-jcap" / "main.tex"
txt = main.read_text(encoding="utf-8")

def wrap_middle(line):
    # Split only on first two ampersands to isolate the middle cell
    parts = line.split('&')
    if len(parts) < 3: 
        return line
    left, mid, right = parts[0], parts[1], '&'.join(parts[2:])
    mid_stripped = mid.strip()
    # if already math, keep
    if mid_stripped.startswith('$') and mid_stripped.endswith('$'):
        return line
    # Heuristic: wrap if mid contains \pm OR is mainly numbers/.,- and spaces
    if r'\pm' in mid_stripped or re.fullmatch(r'[0-9.\sEe+\-*/()]+', mid_stripped):
        mid = ' $' + mid_stripped + '$ '
        return '&'.join([left, mid, right])
    return line

out_lines = []
in_table = False
for line in txt.splitlines(keepends=False):
    if r'\begin{tabular' in line:
        in_table = True
    if in_table and (r'$\tauMem$' in line or r'$\kappaMem$' in line or r'$\wzero$' in line or r'$\Qfact$' in line or r'$C_\mathrm{cal}$' in line or r'\pm' in line):
        out_lines.append(wrap_middle(line))
    else:
        out_lines.append(line)
    if r'\end{tabular}' in line:
        in_table = False

main.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
print("Strict math-wrapping applied to table middle cells.")
PY

# 3) Build again to clear the Missing $ and settle refs
"$ROOT/tools/jcap_build_rescue.sh"
echo "== Strict build complete → paper-jcap/main.pdf"
