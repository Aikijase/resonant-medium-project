#!/usr/bin/env bash
set -euo pipefail

tex=main.tex
base=main
bibfile=refs.bib

echo "=== 0) Files present? =========================="
ls -lh "$tex" || true
ls -lh "$bibfile" || { echo "MISSING: $bibfile"; }

echo; echo "=== 1) Check bibliography commands in $tex ===="
tail -n 200 "$tex" | sed -n '/\\bibliographystyle\|\\bibliography\|\\end{document}/p' || true
have_style=$(grep -c '\\bibliographystyle' "$tex" || true)
have_bib=$(grep -c '\\bibliography' "$tex" || true)
if [[ "$have_style" -eq 0 || "$have_bib" -eq 0 ]]; then
  echo ">> PROBLEM: bibliography commands missing in tex (need \\bibliographystyle{JHEP} and \\bibliography{refs} before \\end{document})"
  if [[ "${1-}" == "--fix-bibcmds" ]]; then
    echo ">> Fixing: inserting just before \\end{document}"
    awk '
      /\\end{document}/ && !s {
        print "\\bibliographystyle{JHEP}\n\\bibliography{refs}"
        s=1
      }
      { print }
    ' "$tex" > .main.tmp && mv .main.tmp "$tex"
  fi
fi

echo; echo "=== 2) Sanity: does $bibfile exist & non-empty? ==="
if [[ ! -s "$bibfile" ]]; then
  echo ">> PROBLEM: $bibfile missing or empty."
  exit 2
fi

echo; echo "=== 3) Clean & compile in the correct order ====="
rm -f ${base}.{aux,bbl,blg,log,out,toc,lof,lot,fls,fdb_latexmk}
pdflatex -interaction=nonstopmode -file-line-error "$tex" >/dev/null || true
echo; echo "--- main.aux: bib handoff ---"
grep -nE '\\bibstyle|\\bibdata|\\citation' ${base}.aux || echo "No bib markers in aux!"

echo; echo "=== 4) Run bibtex and inspect log ==============="
bibtex ${base} >/dev/null || true
echo; sed -n '1,120p' ${base}.blg || true
echo; echo "--- Missing keys (if any) ---"
grep -o "Warning--I didn't find a database entry for .*" ${base}.blg | sort -u || echo "No missing-key warnings."

echo; echo "=== 5) Did bibtex generate any \\bibitem? ======="
if grep -q '\\bibitem' ${base}.bbl 2>/dev/null; then
  echo "OK: \\bibitem entries exist in main.bbl"
  echo; sed -n '1,40p' ${base}.bbl
else
  echo ">> PROBLEM: main.bbl has no \\bibitem entries (bibtex didn’t find usable entries)."
fi

echo; echo "=== 6) Finalize (2x pdflatex) ==================="
pdflatex -interaction=nonstopmode -file-line-error "$tex" >/dev/null || true
pdflatex -interaction=nonstopmode -file-line-error "$tex" >/dev/null || true

echo; echo "=== 7) Quick PDF check ==========================="
if grep -q '\\bibitem' ${base}.bbl 2>/dev/null; then
  echo "If citations still show [?], the issue is *in-text keys* (typos) or Unicode/syntax in $bibfile."
  echo "See missing-key list above and first 120 lines of main.blg."
else
  echo "No \\bibitem -> fix one of: (a) add \\bibliographystyle/\\bibliography in tex, (b) wrong bib filename/path, (c) malformed .bib, (d) biblatex/biber mismatch."
fi
