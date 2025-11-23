#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SRC="$ROOT/paper-jcap"
OUT="$ROOT/paper-jcap-submit"
ZIP="$ROOT/paper-jcap-submit.zip"

rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT/figs"

# Copy essentials
cp "$SRC/main.tex" "$OUT/"
cp "$SRC/refs.bib" "$OUT/" 2>/dev/null || true
cp "$SRC/main.pdf" "$OUT/" 2>/dev/null || true
cp "$SRC/latexmkrc" "$OUT/" 2>/dev/null || true

# Copy figure PDFs
cp "$SRC/figs/"*.pdf "$OUT/figs/" 2>/dev/null || true

# Inject minimal local jcappub.sty (so Overleaf TL2025 compiles)
cat > "$OUT/jcappub.sty" <<'TEX'
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{jcappub}[2025/11/06 local minimal stub for JCAP]
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

# Flip bib style in the SUBMIT COPY to JHEP (Overleaf hosts JHEP.bst)
if grep -q '\\bibliographystyle{' "$OUT/main.tex"; then
  sed -i -E 's#\\bibliographystyle\{[A-Za-z0-9_]+\}#\\bibliographystyle{JHEP}#' "$OUT/main.tex"
else
  echo '\bibliographystyle{JHEP}' >> "$OUT/main.tex"
fi

# Zip the whole submit folder for Overleaf upload
( cd "$ROOT" && zip -r9 "$(basename "$ZIP")" "$(basename "$OUT")" >/dev/null )
echo "Ready to upload: $ZIP"
