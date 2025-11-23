#!/usr/bin/env python3
# Build a Phase-12 one-pager (H-centric but tolerant of your existing plot names)
from pathlib import Path
import pandas as pd

OUTDIR = Path("outputs/phase12")
PLOTDIR = OUTDIR / "plots"
TEX = OUTDIR / "phase12_onepager.tex"

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

# Load alpha_star if available
alpha_star_csv = OUTDIR / "phase12_alpha_star.csv"
if alpha_star_csv.exists():
    df = pd.read_csv(alpha_star_csv)
else:
    # fallback to multichord_wide_alpha_star.csv if your runner produced that
    fallback = PLOTDIR / "multichord_wide_alpha_star.csv"
    df = pd.read_csv(fallback) if fallback.exists() else pd.DataFrame(columns=['br','alpha_star','H_star'])

# Prefer H plots if they exist; otherwise include recent multichord_wide_* plots
preferred = ["H_vs_alpha.png", "H_vs_alpha_by_seed.png"]
figs = [p for p in preferred if (PLOTDIR / p).exists()]

# If not enough figs, add up to 6 newest PNGs in plots/
if len(figs) < 6:
    pngs = sorted(PLOTDIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pngs:
        if p.name not in figs:
            figs.append(p.name)
        if len(figs) >= 6:
            break

tex = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{siunitx}
\sisetup{round-mode=places,round-precision=3}
\title{Phase 12: Multichord Coherence Summary}
\date{}
\begin{document}
\maketitle

\section*{$\alpha^\*$ (median-$H$ optimum)}
\begin{tabular}{@{}lrr@{}}
\toprule
br & $\alpha^\*$ & $H^\*$ \\
\midrule
"""
if len(df):
    has_h = 'H_star' in df.columns
    for _,r in df.iterrows():
        br = int(r['br']) if 'br' in r and pd.notna(r['br']) else -1
        a  = r['alpha_star'] if 'alpha_star' in r else r.get('alpha', '')
        hs = r['H_star'] if has_h else float('nan')
        tex += f"{br} & {a} & {hs:.3f} \\\\\n"
else:
    tex += r"\multicolumn{3}{c}{\emph{No $\alpha^\*$ found. Run the summariser first.}}\\"
tex += r"""\bottomrule
\end{tabular}

\section*{Notes}
We summarise Phase-12 with a lightweight harmony score $H$ and show diagnostic plots from the run (H-centric if available, otherwise the latest multichord figures).

\section*{Diagnostics}
"""

for p in figs:
    tex += r"\begin{center}\includegraphics[width=0.9\linewidth]{plots/"+p+r"}\end{center}"+"\n"

tex += r"""
\end{document}
"""
TEX.write_text(tex)
print(str(TEX))
