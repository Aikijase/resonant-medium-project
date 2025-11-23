#!/usr/bin/env bash
set -euo pipefail
cd outputs/phase18
pdflatex -halt-on-error -interaction=nonstopmode p18_report.tex >/dev/null
pdflatex -halt-on-error -interaction=nonstopmode p18_report.tex >/dev/null
echo "✅ Wrote: $(realpath p18_report.pdf)"
