#!/usr/bin/env bash
set -euo pipefail
cd outputs/phase16
pdflatex -halt-on-error -interaction=nonstopmode p16_report.tex >/dev/null
pdflatex -halt-on-error -interaction=nonstopmode p16_report.tex >/dev/null
echo "✅ Wrote: $(realpath p16_report.pdf)"
