k#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PAPER="$ROOT/paper-jcap"
MAIN="$PAPER/main.tex"
FIGS="$PAPER/figs"
FIT="$ROOT/outputs/phase20/p20_fit.json"

echo "== JCAP RESCUE BUILD =="

[[ -f "$MAIN" ]] || { echo "ERROR: $MAIN not found. Run tools/jcap_init.sh first."; exit 1; }
mkdir -p "$FIGS"

# --- Ensure figures: PNG->PDF if needed ---
need_pdf () {
  local base="$1"
  if [[ ! -f "$FIGS/$base.pdf" ]]; then
    if [[ -f "$ROOT/outputs/phase20/$base.pdf" ]]; then
      cp "$ROOT/outputs/phase20/$base.pdf" "$FIGS/$base.pdf"
      echo "copied $base.pdf"
    elif [[ -f "$ROOT/outputs/phase20/$base.png" ]]; then
      if command -v img2pdf >/dev/null 2>&1; then
        img2pdf "$ROOT/outputs/phase20/$base.png" -o "$FIGS/$base.pdf"
        echo "converted $base.png -> $base.pdf"
      else
        echo "WARN: install img2pdf to convert PNG → PDF"; fi
    else
      echo "WARN: missing figure $base.[pdf|png]"
    fi
  fi
}
need_pdf p20_plot_robust
need_pdf p20_plot_kphi

# --- Make a local jcappub stub only if jcappub.sty is missing ---
LOCAL_STUB=0
if ! kpsewhich jcappub.sty >/dev/null 2>&1 || [[ -z "$(kpsewhich jcappub.sty || true)" ]]; then
  LOCAL_STUB=1
  cat > "$PAPER/jcappub.sty" <<'TEX'
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{jcappub}[2025/11/06 local minimal stub]
\RequirePackage{ifthen}
\RequirePackage{etoolbox}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{hyperref}
\RequirePackage{booktabs}
\providecommand{\flushbottom}{}
\newcommand{\emailAdd}[1]{\thanks{\href{mailto:#1}{#1}}}
\newcommand{\affiliation}[2][]{\newcommand{\tmpaffil}{#2}}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}
\newcommand{\arxivnumber}[1]{\gdef\@arxiv{#1}}
\newcommand{\acknowledgments}{\section*{Acknowledgments}}
\makeatletter
\AtBeginDocument{\providecommand{\@keywords}{}\providecommand{\@arxiv}{}}
\pretocmd{\maketitle}{%
  \begin{center}
  \ifx\@arxiv\@empty\else \vspace{-0.5\baselineskip}{\small arXiv:\@arxiv}\par\vspace{0.5\baselineskip}\fi
  \ifx\@keywords\@empty\else {\small \textbf{Keywords:} \@keywords}\par\vspace{0.5\baselineskip}\fi
  \end{center}
}{}{}
\makeatother
TEX
  echo "using local jcappub stub"
fi

# --- Python fixer/builder (fully replaces bad helper blocks, fills table, normalizes for stub) ---
python3 - <<'PY'
import re, json, pathlib, sys, os

ROOT = pathlib.Path(os.environ.get("ROOT", ".")).resolve()
PAPER = ROOT / "paper-jcap"
MAIN = PAPER / "main.tex"
FIT  = ROOT / "outputs/phase20/p20_fit.json"
use_stub = os.environ.get("LOCAL_STUB","0") == "1"

txt = MAIN.read_text(encoding="utf-8")

def fm(x):
    m, s = x["mean"], x["sd"]
    if s >= 0.1: return f"{m:.2f} \\pm {s:.2f}"
    if s >= 0.01: return f"{m:.3f} \\pm {s:.3f}"
    return f"{m:.4f} \\pm {s:.4f}"

# 1) Remove any previous broken helper block (with literal '\n')
txt = re.sub(r"\n?\\n\s*% Local math helpers.*?(?=\n\s*\n|\\section|\\begin\{document\}|\\maketitle)", "\n", txt, flags=re.S)

# 2) Insert a clean helper block with real newlines (once), right after the last \usepackage line
helper = (
    "% Local math helpers (for local build)\n"
    "\\providecommand{\\abs}[1]{\\left\\lvert#1\\right\\rvert}\n"
    "\\providecommand{\\degree}{\\ensuremath{^{\\circ}}}\n"
)
upkgs = list(re.finditer(r'\\usepackage\{[^\}]+\}.*\n', txt))
if upkgs:
    last = upkgs[-1].end()
    txt = txt[:last] + helper + txt[last:]
elif "\\documentclass" in txt:
    dc = re.search(r'\\documentclass[^\n]*\n', txt)
    pos = dc.end() if dc else 0
    txt = txt[:pos] + helper + txt[pos:]
else:
    txt = helper + txt

# 3) If using stub, normalize author/affiliation/email and move abstract inside document
if use_stub:
    ma = re.search(r'\\author\[[^\]]*\]\{([^}]*)\}', txt)
    name = ma.group(1).strip() if ma else None
    maff = re.search(r'\\affiliation\[[^\]]*\]\{([^}]*)\}', txt)
    aff  = maff.group(1).strip() if maff else ""
    mm   = re.search(r'\\emailAdd\{([^}]*)\}', txt)
    mail = mm.group(1).strip() if mm else None

    if ma:
        new = "\\author{" + name
        if mail: new += "\\thanks{\\href{mailto:" + mail + "}{" + mail + "}}"
        if aff:  new += "\\\\\\small " + aff
        new += "}\n"
        txt = re.sub(r'\\author\[[^\]]*\]\{[^}]*\}\s*', new, txt)
    txt = re.sub(r'\\affiliation\[[^\]]*\]\{[^}]*\}\s*', '', txt)
    txt = re.sub(r'\\emailAdd\{[^}]*\}\s*', '', txt)

    # Move \abstract{...} into environment after \maketitle
    if r'\begin{abstract}' not in txt:
        mabs = re.search(r'\\abstract\{(.*?)\}', txt, flags=re.S)
        if mabs:
            ab = mabs.group(1).strip()
            txt = txt.replace(mabs.group(0), '')
            mmt = re.search(r'(\\maketitle)', txt)
            if mmt:
                pos = mmt.end()
                txt = txt[:pos] + "\n\\begin{abstract}\n" + ab + "\n\\end{abstract}\n" + txt[pos:]
    # Local-friendly bib style
    txt = re.sub(r'\\bibliographystyle\{[^}]+\}', r'\\bibliographystyle{unsrt}', txt)

# 4) Fill table values if fit json exists
if FIT.exists():
    d = json.loads(FIT.read_text(encoding="utf-8"))
    repls = [
        (r'(\$\\tauMem\$\s*&\s*)<NUM>',           r'\1' + fm(d['tau'])),
        (r'(\$\\kappaMem\$\s*&\s*)<NUM>',         r'\1' + fm(d['kappa'])),
        (r'(\$\\wzero\$\s*&\s*)<NUM>',            r'\1' + fm(d['omega0'])),
        (r'(\$\\Qfact\$\s*&\s*)<NUM>',            r'\1' + fm(d['Q'])),
        (r'(\$C_\\mathrm\{cal\}\$\s*&\s*)<NUM>',  r'\1' + fm(d['C_cal'])),
    ]
    for pat, rep in repls:
        txt = re.sub(pat, lambda m, rep=rep: m.group(1) + rep, txt)

MAIN.write_text(txt, encoding="utf-8")
print("main.tex repaired + filled")
PY

# --- Unicode/typography sweep (safe) ---
fixu () {
  local pat="$1" rep="$2"
  grep -RIl --null "$pat" "$PAPER" | while IFS= read -r -d '' f; do sed -i "s/$pat/$rep/g" "$f"; done || true
}
fixu "≈" "\\\\approx{}"
fixu "±" "\\\\pm{}"
fixu "–" "--"
fixu "—" "---"
fixu "°" "\\\\degree{}"
fixu "“" "``"
fixu "”" "''"
fixu "’" "'"

# --- Build ---
pushd "$PAPER" >/dev/null
if ! command -v latexmk >/dev/null 2>&1; then
  echo "WARN: latexmk not installed; trying pdflatex/bibtex fallback"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
  bibtex main || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
else
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi
popd >/dev/null

echo "== Done → $PAPER/main.pdf"

