#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PAPER_DIR="$ROOT/paper-jcap"
FIG_DIR="$PAPER_DIR/figs"

mkdir -p "$PAPER_DIR" "$FIG_DIR"

# Copy (or fallback) figures
for f in p20_plot_robust.pdf p20_plot_kphi.pdf; do
  if [[ -f "$ROOT/outputs/phase20/$f" ]]; then
    cp "$ROOT/outputs/phase20/$f" "$FIG_DIR/$f"
  else
    echo "WARN: $f not found in outputs/phase20; leaving placeholder reference."
  fi
done

# Write main.tex (JCAP clean skeleton)
cat > "$PAPER_DIR/main.tex" <<'TEX'
\documentclass[a4paper,11pt]{article}
\usepackage{jcappub}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{siunitx}
\usepackage{graphicx,booktabs,multirow}
\usepackage{hyperref}

\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}
\sisetup{detect-weight=true,detect-family=true}

% --- Symbols ---
\newcommand{\tauMem}{\tau}
\newcommand{\kappaMem}{\kappa}
\newcommand{\wzero}{\omega_0}
\newcommand{\Qfact}{Q}
\newcommand{\fsig}{f\sigma_8}

\title{The Echo Equation in Resonant Medium Cosmology}
\author[a]{Jason Watts}
\affiliation[a]{Resonant Medium Project, Bendigo, Australia}
\emailAdd{your.email@example.org}
\abstract{
We present a finite-memory, single-frequency kernel for late-time cosmology---the Echo Equation---and show how its four parameters $(\tau,\kappa,\wzero,Q)$ map to background and linear-growth observables. In the $\tau,\kappa\to 0$ limit the model reduces to $\Lambda$CDM. Using Phase--20 runs, we illustrate compatibility at comparable fit quality and specify a one-frequency, cross-probe joint test (RSD and $g\times\kappa$). The model is falsified if a common $\wzero$ is not recovered across growth probes or if amplitudes vanish at the shared frequency given survey depth.
}
\keywords{cosmology of theories beyond $\Lambda$CDM, cosmological perturbation theory, modified gravity}
\arxivnumber{XXXX.XXXXX}

\begin{document}
\maketitle
\flushbottom

\section{Model}
We add a causal, exponentially-damped, single-frequency memory term. In the $\tauMem\!\to\!0$, $\kappaMem\!\to\!0$ limit, the model reduces to $\Lambda$CDM. The residual frequency is $\wzero$ (units: rad per e-fold):
\begin{equation}\label{eq:kernel}
K(\Delta N) = \tauMem\, e^{-\Delta N/\Qfact}\cos\!\big(\wzero\,\Delta N\big)
             + \kappaMem\, e^{-\Delta N/\Qfact}\sin\!\big(\wzero\,\Delta N\big),
\end{equation}
with $\Delta N=\ln a(t)-\ln a(t')$. For small amplitudes, the induced growth residual $R(N)$ is linear in $\kappaMem$; empirically $A \approx C_\mathrm{cal}\,\kappaMem$ with $C_\mathrm{cal}$ reported in Table~\ref{tab:post}.

\begin{center}
\fbox{\parbox{0.92\linewidth}{
\textbf{Joint test (one frequency):} Fit a common $\wzero$ to RSD and $g\times\kappa$; allow sector amplitudes $A_{\mathrm{RSD}}, A_{g\kappa}$. Decision rule: detect $A\neq 0$ at the \emph{shared} $\wzero$; else quote a 95\% CL bound on $\abs{\kappaMem}$ via $A\propto\kappaMem$.
}}
\end{center}

\section{Results}
\begin{figure}[t]
  \centering
  \includegraphics[width=0.82\linewidth]{figs/p20_plot_robust.pdf}
  \caption{Representative Phase--20 residuals vs.\ $\ln a$ at the recovered $\wzero$. Shaded band: $1\sigma$ posterior.}
  \label{fig:residuals}
\end{figure}

\begin{table}[t]
\centering
\caption{Posterior means $\pm 1\sigma$ (Phase--20 lock). Numerical values auto-filled by \texttt{jcap\_fill\_posteriors.py}.}
\label{tab:post}
\begin{tabular}{lcc}
\toprule
Parameter & Mean $\pm$ SD & Units \\
\midrule
$\tauMem$   & <NUM> & dimensionless \\
$\kappaMem$ & <NUM> & dimensionless \\
$\wzero$    & <NUM> & rad/e-fold \\
$\Qfact$    & <NUM> & dimensionless \\
$C_\mathrm{cal}$ & <NUM> & (amplitude per $\kappa$) \\
\bottomrule
\end{tabular}
\end{table}

\section{Discussion and $\Lambda$CDM limit}
For $\tauMem,\kappaMem\to 0$, Eq.~\eqref{eq:kernel} vanishes and $\Lambda$CDM is recovered. For concreteness, if $\wzero=\SI{1.6}{rad/e\mbox{-}fold}$, the half-period scale is $\Delta N = \pi/\wzero \approx 1.96$ e-folds.

\section{Data and code availability}
All Phase--20 artifacts and scripts are available in the project repository (tag: \texttt{phase20\_lock}).

\acknowledgments
Thanks to Mal, Taya, and collaborators for patience and feedback.

\bibliographystyle{JHEP}
\bibliography{refs}
\end{document}
TEX

# Minimal refs.bib
cat > "$PAPER_DIR/refs.bib" <<'BIB'
@article{DeLeo2025,
  author = {De Leo, Chiara and Martinelli, Matteo and D’Agostino, Rocco and Gianfagna, Giulia and Martins, C.J.A.P.},
  title = {Distinguishing distance duality breaking models using electromagnetic and gravitational waves measurements},
  journal = {JCAP},
  year = {2025},
  number = {11},
  pages = {001},
  doi = {10.1088/1475-7516/2025/11/001},
  eprint = {2505.13613},
  archivePrefix = {arXiv}
}
@article{Planck2018,
  author = {Planck Collaboration},
  title = {Planck 2018 results. VI. Cosmological parameters},
  journal = {A\&A},
  year = {2020},
  volume = {641},
  pages = {A6},
  eprint = {1807.06209}
}
@article{PantheonPlus,
  author = {Scolnic, D. and others},
  title = {The Pantheon+ Analysis of Type Ia Supernovae},
  journal = {ApJ},
  year = {2022},
  eprint = {2112.03863}
}
@article{Etherington1933,
  author = {Etherington, I. M. H.},
  title = {On the definition of distance in general relativity},
  journal = {Phil. Mag.},
  year = {1933},
  volume = {15},
  pages = {761}
}
BIB

# latexmk config for local builds
cat > "$PAPER_DIR/latexmkrc" <<'RC'
$pdf_mode = 1;
$interaction = 'nonstopmode';
$halt_on_error = 1;
$silent = 0;
add_cus_dep('bib','bbl',0,'do_bibtex');
sub do_bibtex { system("bibtex '$_[0]'"); }
RC

# Makefile convenience
cat > "$PAPER_DIR/Makefile" <<'MK'
all: fix fill build

fix:
	../tools/jcap_fix_unicode.sh

fill:
	python3 ../tools/jcap_fill_posteriors.py --fit ../outputs/phase20/p20_fit.json --tex main.tex --out main.tex

build:
	latexmk -pdf main.tex

clean:
	latexmk -C
MK

echo "JCAP paper initialized at: $PAPER_DIR"
