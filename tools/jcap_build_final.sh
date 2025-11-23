#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PAPER="$ROOT/paper-jcap"
MAIN="$PAPER/main.tex"
FIGS="$PAPER/figs"
FIT="$ROOT/outputs/phase20/p20_fit.json"

[[ -f "$MAIN" ]] || { echo "ERROR: $MAIN not found. Run tools/jcap_init.sh first."; exit 1; }
mkdir -p "$FIGS"

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
        echo "WARN: install img2pdf to convert PNG → PDF"
      fi
    else
      echo "WARN: missing figure $base.[pdf|png]"
    fi
  fi
}
need_pdf p20_plot_robust
need_pdf p20_plot_kphi

# Local jcappub stub only if missing
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

# Python: inject helpers (non-destructive), stub-safe author/abstract, fill values as math
python3 - "$MAIN" "$FIT" "$LOCAL_STUB" <<'PY'
import re, json, pathlib, sys

main_path = pathlib.Path(sys.argv[1])
fit_path  = pathlib.Path(sys.argv[2])
use_stub  = (sys.argv[3] == "1")
txt = main_path.read_text(encoding="utf-8")

def fm_math(x):
    m, s = x["mean"], x["sd"]
    if s >= 0.1: sf, mf = f"{s:.2f}", f"{m:.2f}"
    elif s >= 0.01: sf, mf = f"{s:.3f}", f"{m:.3f}"
    else: sf, mf = f"{s:.4f}", f"{m:.4f}"
    return f"${mf} \\\\pm {sf}$"

# Helpers once, real newlines
if "\\providecommand{\\abs}" not in txt or "\\providecommand{\\degree}" not in txt:
    helper = (
        "\n% Local math helpers (for local build)\n"
        "\\providecommand{\\abs}[1]{\\left\\lvert#1\\right\\rvert}\n"
        "\\providecommand{\\degree}{\\ensuremath{^{\\circ}}}\n"
    )
    m = list(re.finditer(r'(\\usepackage\{[^\}]+\}.*\n)', txt))
    if m:
        txt = txt[:m[-1].end()] + helper + txt[m[-1].end():]
    else:
        d = re.search(r'(\\documentclass[^\n]*\n)', txt)
        pos = d.end() if d else 0
        txt = txt[:pos] + helper + txt[pos:]

# Stub compatibility for author/affiliation/email + abstract
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

    if r'\begin{abstract}' not in txt:
        mabs = re.search(r'\\abstract\{(.*?)\}', txt, flags=re.S)
        if mabs:
            ab = mabs.group(1).strip()
            txt = txt.replace(mabs.group(0), '')
            mmt = re.search(r'(\\maketitle)', txt)
            if mmt:
                pos = mmt.end()
                txt = txt[:pos] + "\n\\begin{abstract}\n" + ab + "\n\\end{abstract}\n" + txt[pos:]
    txt = re.sub(r'\\bibliographystyle\{[^}]+\}', r'\\bibliographystyle{unsrt}', txt)

# Header middle cell → single math block
txt = re.sub(r'(&\s*)(Mean\s*±\s*SD|\$Mean\s*\\pm\s*SD\$)(\s*&)', r'\1$Mean \\pm SD$\3', txt)

# Fill numeric cells as math
if fit_path.exists():
    d = json.loads(fit_path.read_text(encoding="utf-8"))
    repls = [
        (r'(\$\\tauMem\$\s*&\s*)<NUM>',           r'\1' + fm_math(d['tau'])),
        (r'(\$\\kappaMem\$\s*&\s*)<NUM>',         r'\1' + fm_math(d['kappa'])),
        (r'(\$\\wzero\$\s*&\s*)<NUM>',            r'\1' + fm_math(d['omega0'])),
        (r'(\$\\Qfact\$\s*&\s*)<NUM>',            r'\1' + fm_math(d['Q'])),
        (r'(\$C_\\mathrm\{cal\}\$\s*&\s*)<NUM>',  r'\1' + fm_math(d['C_cal'])),
    ]
    for pat, rep in repls:
        txt = re.sub(pat, lambda m, rep=rep: m.group(1) + rep, txt)

main_path.write_text(txt, encoding="utf-8")
print("main.tex updated safely.")
PY

# Typography sweep (safe)
fixu () { local a="$1" b="$2"; grep -RIl --null "$a" "$PAPER" | while IFS= read -r -d '' f; do sed -i "s/$a/$b/g" "$f"; done || true; }
fixu "≈" "\\\\approx{}"
fixu "±" "\\\\pm{}"
fixu "–" "--"
fixu "—" "---"
fixu "°" "\\\\degree{}"
fixu "“" "``"
fixu "”" "''"
fixu "’" "'"

# Build
pushd "$PAPER" >/dev/null
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  echo "WARN: install latexmk (sudo apt-get install -y latexmk texlive-latex-extra texlive-fonts-recommended)"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
  bibtex main || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
fi
popd >/dev/null
