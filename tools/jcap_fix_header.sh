#!/usr/bin/env bash
set -euo pipefail
MAIN="paper-jcap/main.tex"
# Normalize the header middle cell to a single math block: $Mean \pm SD$
# This catches any of these variants: "Mean ± SD", "$Mean $\pm $ SD$", etc.
sed -i -E \
  -e 's/(&[[:space:]]*)Mean[[:space:]]*±[[:space:]]*SD([[:space:]]*&)/\1$Mean \\pm SD$\2/' \
  -e 's/(&[[:space:]]*)\$Mean[[:space:]]*\$\\pm[[:space:]]*\$[[:space:]]*SD\$(\s*&)/\1$Mean \\pm SD$\2/' \
  -e 's/(&[[:space:]]*)\$Mean[[:space:]]*\\pm[[:space:]]*SD\$(\s*&)/\1$Mean \\pm SD$\2/' \
  "$MAIN"
echo "✔ Header cell normalized to: $Mean \\pm SD$"
