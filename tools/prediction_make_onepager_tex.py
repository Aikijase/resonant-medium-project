#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pred = root / "outputs" / "predictions"
pred.mkdir(parents=True, exist_ok=True)

def load_params(p):
    if not p.exists():
        return {"A":1.0,"tau_m":1.0,"omega":1.0,"phi":0.0,"chi2":0.0}
    j = json.loads(p.read_text())
    g = j.get("params", j)
    return {
        "A": float(g.get("A",1.0)),
        "tau_m": float(g.get("tau_m", g.get("tau",1.0))),
        "omega": float(g.get("omega",1.0)),
        "phi": float(g.get("phi",0.0)),
        "chi2": float(j.get("chi2", j.get("chi_sq",0.0))),
    }

p20 = load_params(root/"outputs/phase20/p20.fit.json")
gb  = 10.0
gb_csv = root/"outputs/phase21/p21_guardbands.csv"
if gb_csv.exists():
    try:
        for line in gb_csv.read_text().splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts)>=2 and parts[0].lower().startswith("omega_pct_window"):
                gb = abs(float(parts[1]))*100.0
                break
    except:
        pass

tex = rf"""% Auto-generated: predictions_onepager.tex (Lock)
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{graphicx,amsmath}}
\title{{Resonant Medium — Predictions (Lock One-Pager)}}
\date{{\today}}
\begin{{document}}\maketitle

\textbf{{Kernel (Phase-20 effective)}}: 
$A={p20['A']:.3g}$, $\tau_m={p20['tau_m']:.3g}$, $\omega={p20['omega']:.3g}$, $\phi={p20['phi']:.3g}$; 
$\chi^2={p20['chi2']:.3f}$. Guardband: $\pm {gb:.1f}\%$ around $\omega$.

\section*{{P1: Low-$\omega$ lensing residual (template)}}
Defines residual$(\ell)=A e^{{-t(\ell)/\tau_m}}\cos(\omega t(\ell)+\phi)$ with fixed $t(\ell)=3\times10^{{-3}}\ell$.
\begin{{center}}
\includegraphics[width=0.9\linewidth]{{plot_lensing_template.png}}
\end{{center}}

\section*{{P2: BAO phase tweak (proxy; ablation)}}
with\_memory $=$ base$(z)+0.01\cdot A e^{{-z/\tau_m}}\cos(\omega z+\phi)$; ablation $\tau_m\to 0$ removes the tweak.
\begin{{center}}
\includegraphics[width=0.9\linewidth]{{plot_bao_tweak.png}}
\end{{center}}

\end{{document}}
"""
(pred/"predictions_onepager.tex").write_text(tex)
print("[pred] wrote:", pred/"predictions_onepager.tex")
