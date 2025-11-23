#!/usr/bin/env bash
set -euo pipefail
MAIN="paper-jcap/main.tex"
# Wrap the middle cell (between & … &) in $…$ for the rows we auto-fill
# Matches lines starting with a parameter name like "$\tauMem$ & 0.300 \pm 0.040 & ..."
sed -i -E \
  -e 's/^(\s*\$\\tauMem\$\s*&\s*)([^&]+)(\s*&)/\1$\2$\3/' \
  -e 's/^(\s*\$\\kappaMem\$\s*&\s*)([^&]+)(\s*&)/\1$\2$\3/' \
  -e 's/^(\s*\$\\wzero\$\s*&\s*)([^&]+)(\s*&)/\1$\2$\3/' \
  -e 's/^(\s*\$\\Qfact\$\s*&\s*)([^&]+)(\s*&)/\1$\2$\3/' \
  -e 's/^(\s*\$C_\\mathrm\{cal\}\$\s*&\s*)([^&]+)(\s*&)/\1$\2$\3/' \
  "$MAIN"
echo "Wrapped table numeric cells in math mode."
