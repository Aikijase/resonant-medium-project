#!/usr/bin/env bash
set -e

# Output dir
OUTDIR="outputs/phase20"
PREFIX="p20"

# Make sure the report exists
TEXFILE="${OUTDIR}/${PREFIX}_report.tex"
PDFFILE="${OUTDIR}/${PREFIX}_report.pdf"

# Build minimal LaTeX report
cat <<'EOF' > "$TEXFILE"
\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}

\title{Phase 20 — Edge Predictor and Guardband}
\author{Resonant Medium Project}
\date{\today}

\begin{document}
\maketitle

\section*{Overview}
This report summarizes the predictive edge model for $\omega^2 \rightarrow K_\phi$, with curvature and robustness-based guardband warnings.

\section*{Predictions Table}
See \texttt{p20\_predictions.csv} for full numeric sweep data.

\section*{Plots}
\begin{center}
\includegraphics[width=0.9\textwidth]{p20_plot_kphi.png}
\end{center}

\begin{center}
\includegraphics[width=0.9\textwidth]{p20_plot_robust.png}
\end{center}

\end{document}
EOF

# Build PDF
pdflatex -output-directory="$OUTDIR" "$TEXFILE" >/dev/null 2>&1
pdflatex -output-directory="$OUTDIR" "$TEXFILE" >/dev/null 2>&1

echo "✅ Wrote: $PDFFILE"
