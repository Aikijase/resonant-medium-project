#!/usr/bin/env bash
set -euo pipefail
cd outputs/phase19
pdflatex -halt-on-error -interaction=nonstopmode p19_report.tex >/dev/null
pdflatex -halt-on-error -interaction=nonstopmode p19_report.tex >/dev/null
echo "✅ Wrote: $(realpath p19_report.pdf)"
