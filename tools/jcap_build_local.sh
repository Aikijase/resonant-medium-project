#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PAPER="$ROOT/paper-jcap"
MAIN="$PAPER/main.tex"
FIGS="$PAPER/figs"
FIT="$ROOT/outputs/phase20/p20_fit.json"

echo "== JCAP one-shot build (local) =="

[[ -f "$MAIN" ]] || { echo "ERROR: $MAIN not found. Run tools/jcap_init.sh first."; exit 1; }
mkdir -p "$FIGS"

need_pdf() {
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
        echo "WARN: img2pdf not installed; cannot convert $base.png. Install with: sudo apt-get install -y img2pdf"
      fi
    else
      echo "WARN: missing figure for $base (neither PDF nor PNG found)."
    fi
  fi
}
need_pdf p20_plot_robust
need_pdf p20_plot_kphi

# If jcappub.sty is missing, write a minimal local stub so we can compile
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
  echo "using local jcappub stub at paper-jcap/jcappub.sty"
fi

# Python: (a) make local-stub-safe author/email/abstract; (b) define helpers; (c) fill table values
python3 - "$MAIN" "$FIT" "$LOCAL_STUB" <<'PY'
import sys, re, json, pathlib

main_path = pathlib.Path(sys.argv[1])
fit_path  = pathlib.Path(sys.argv[2])
use_stub  = (sys.argv[3] == "1")

txt = main_path.read_text(encoding="utf-8")

def fm(x):
    m, s = x["mean"], x["sd"]
    if s >= 0.1: return f"{m:.2f} \\pm {s:.2f}"
    if s >= 0.01: return f"{m:.3f} \\pm {s:.3f}"
    return f"{m:.4f} \\pm {s:.4f}"

# Insert local helper cmds once (real newlines)
if r'\providecommand{\abs}' not in txt or r'\providecommand{\degree}' not in txt:
    helper = (
        "\n% Local math helpers (for local build)\n"
        "\\providecommand{\\abs}[1]{\\left\\lvert#1\\right\\rvert}\n"
        "\\providecommand{\\degree}{\\ensuremath{^{\\circ}}}\n"
    )
    # place after last \usepackage line
    m = list(re.finditer(r'(\\usepackage\{[^\}]+\}.*\n)', txt))
    if m:
        last = m[-1]
        txt = txt[:last.end()] + helper + txt[last.end():]
    else:
        # fallback: put after \documentclass line
        m2 = re.search(r'(\\documentclass[^\n]*\n)', txt)
        if m2:
            txt = txt[:m2.end()] + helper + txt[m2.end():]
        else:
            txt = helper + txt

# If using stub, normalize author/affiliation/email and move abstract into document
if use_stub:
    # author with optional [..]
    m_auth = re.search(r'\\author\[[^\]]*\]\{([^}]*)\}', txt)
    name = m_auth.group(1).strip() if m_auth else None
    m_aff  = re.search(r'\\affiliation\[[^\]]*\]\{([^}]*)\}', txt)
    aff    = m_aff.group(1).strip() if m_aff else ""
    m_mail = re.search(r'\\emailAdd\{([^}]*)\}', txt)
    mail   = m_mail.group(1).strip() if m_mail else None

    if m_auth:
        auth_new = "\\author{" + name
        if mail:
            auth_new += "\\thanks{\\href{mailto:" + mail + "}{" + mail + "}}"
        if aff:
            auth_new += "\\\\\\small " + aff
        auth_new += "}\n"
        txt = re.sub(r'\\author\[[^\]]*\]\{[^}]*\}\s*', auth_new, txt)
    txt = re.sub(r'\\affiliation\[[^\]]*\]\{[^}]*\}\s*', '', txt)
    txt = re.sub(r'\\emailAdd\{[^}]*\}\s*', '', txt)

    if r'\begin{abstract}' not in txt:
        m_abs = re.search(r'\\abstract\{(.*?)\}', txt, flags=re.S)
        if m_abs:
            abs_text = m_abs.group(1).strip()
            txt = txt.replace(m_abs.group(0), '')
            m_mt = re.search(r'(\\maketitle)', txt)
            if m_mt:
                pos = m_mt.end()
                txt = txt[:pos] + "\n\\begin{abstract}\n" + abs_text + "\n\\end{abstract}\n" + txt[pos:]

    txt = re.sub(r'\\bibliographystyle\{[^}]+\}', r'\\bibliographystyle{unsrt}', txt)

# Fill table numbers
if fit_path.exists():
    d = json.loads(fit_path.read_text(encoding="utf-8"))
    repls = [
        (r'(\$\\tauMem\$\s*&\s*)<NUM>',           r'\1' + fm(d['tau'])),
        (r'(\$\\kappaMem\$\s*&\s*)<NUM>',         r'\1' + fm(d['kappa'])),
        (r'(\$\\wzero\$\s*&\s*)<NUM>',            r'\1' + fm(d['omega0'])),
        (r'(\$\\Qfact\$\s*&\s*)<NUM>',            r'\1' + fm(d['Q'])),
        (r'(\$C_\\mathrm\{cal\}\$\s*&\s*)<NUM>',  r'\1' + fm(d['C_cal'])),
    ]
    for pat, rep in repls:
        txt = re.sub(pat, lambda m, rep=rep: m.group(1) + rep, txt)

main_path.write_text(txt, encoding="utf-8")
print("main.tex updated: helpers/author/abstract/bib (if stub) + table values")
PY

# Unicode / typography sweep
fixu() {
  local pat="$1" rep="$2"
  grep -RIl --null "$pat" "$PAPER" | while IFS= read -r -d '' f; do
    sed -i "s/$pat/$rep/g" "$f"
  done || true
}
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
if ! command -v latexmk >/dev/null 2>&1; then
  echo "WARN: latexmk not installed. Install: sudo apt-get install -y latexmk texlive-latex-extra texlive-fonts-recommended"
  echo "Trying pdflatex directly..."
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
  bibtex main || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
  pdflatex -interaction=nonstopmode -halt-on-error main.tex || true
else
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi
popd >/dev/null

echo "== Done → $PAPER/main.pdf"
