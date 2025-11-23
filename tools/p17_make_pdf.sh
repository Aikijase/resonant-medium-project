#!/usr/bin/env bash
set -euo pipefail
cd outputs/phase17
pdflatex -halt-on-error -interaction=nonstopmode p17_report.tex >/dev/null
pdflatex -halt-on-error -interaction=nonstopmode p17_report.tex >/dev/null
echo "✅ Wrote: $(realpath p17_report.pdf)"
