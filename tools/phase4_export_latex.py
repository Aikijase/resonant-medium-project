#!/usr/bin/env python3
"""
Export LaTeX tables for Phase-4 g×κ.

Inputs:
  - outputs/phase4/gk_score.json        (single-catalog)
  - outputs/phase4/gk_multi.json        (multi-bin)

Outputs:
  - outputs/phase4/gk_table.tex
  - outputs/phase4/gk_multi_table.tex
"""
import json, os, math, pathlib

OUTDIR = "outputs/phase4"
SINGLE_JSON = f"{OUTDIR}/gk_score.json"
MULTI_JSON  = f"{OUTDIR}/gk_multi.json"

def fmt(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except: return "—"

def latex_escape(s: str) -> str:
    return (s.replace("&","\\&")
             .replace("%","\\%")
             .replace("$","\\$")
             .replace("#","\\#")
             .replace("_","\\_")
             .replace("{","\\{")
             .replace("}","\\}")
             .replace("~","\\textasciitilde{}")
             .replace("^","\\textasciicircum{}"))

def write_single():
    if not os.path.exists(SINGLE_JSON): return
    J = json.load(open(SINGLE_JSON))
    lines = []
    lines.append("% Auto-generated Phase-4 g×κ (single) table")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Catalog & $z_\\mathrm{eff}$ & $A_{\\rm lit}\\pm\\sigma$ & Resid. ($\\sigma$) \\\\")
    lines.append("\\midrule")
    for d in J.get("details", []):
        name = latex_escape(d["name"])
        zeff = fmt(d["z_eff"],2)
        Alit = fmt(d["A_lit"]); sig = fmt(d["sigma_lit"])
        resid = f"{d['resid_sigma_model']:+.2f}"
        lines.append(f"{name} & {zeff} & {Alit}$\\pm${sig} & {resid} \\\\")
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{4}}{{r}}{{"
                 f"$\\hat A={fmt(J['A_hat'])}\\pm{fmt(J['sigma_A_hat'])}$, "
                 f"$A_\\mathrm{{pred}}={fmt(J['scale_model'])}$, "
                 f"$\\langle|r|\\rangle\\approx{math.sqrt(J['chi2']/max(J['n'],1)):.2f}\\,\\sigma$, "
                 f"$\\Delta\\mathrm{{AIC}}={fmt(J['AIC']['delta'])}$, WWI={J.get('WWI')} }}")
    lines.append("\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Phase-4 $g\\times\\kappa$ (single-catalog) shape-only summary.}")
    lines.append("\\end{table}")
    pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    open(f"{OUTDIR}/gk_table.tex","w").write("\n".join(lines)+"\n")
    print(f"Wrote {OUTDIR}/gk_table.tex")

def write_multi():
    if not os.path.exists(MULTI_JSON): return
    J = json.load(open(MULTI_JSON))
    # Per-survey summary
    lines = []
    lines.append("% Auto-generated Phase-4 g×κ (multi-bin) tables")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Survey & $n$ & $\\hat A\\pm\\sigma$ & $\\langle|r|\\rangle\\ (\\sigma)$ & $\\Delta\\mathrm{AIC}$ & WWI \\\\")
    lines.append("\\midrule")
    for sname, S in sorted(J["surveys"].items()):
        mean_s = math.sqrt(S["chi2_pred"]/max(S["n"],1))
        lines.append(f"{latex_escape(sname)} & {S['n']} & {fmt(S['A_hat'])}$\\pm${fmt(S['sigma_A_hat'])} & {mean_s:.2f} & {fmt(S['AIC']['delta'])} & {S['WWI']} \\\\")
    lines.append("\\midrule")
    mean = math.sqrt(J["chi2_pred"]/max(J["n"],1))
    lines.append(f"\\multicolumn{{6}}{{r}}{{Overall: $\\hat A={fmt(J['A_hat'])}\\pm{fmt(J['sigma_A_hat'])}$, "
                 f"$A_\\mathrm{{pred}}={fmt(J['scale_model'])}$, "
                 f"$\\langle|r|\\rangle\\approx{mean:.2f}\\,\\sigma$, "
                 f"$\\Delta\\mathrm{{AIC}}={fmt(J['AIC']['delta'])}$, WWI={J['WWI']}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Phase-4 $g\\times\\kappa$ (multi-bin) per-survey summary.}")
    lines.append("\\end{table}")
    lines.append("")

    # Per-bin table
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcrrr}")
    lines.append("\\toprule")
    lines.append("Survey & Bin & $z_\\mathrm{eff}$ & $A_{\\rm lit}\\pm\\sigma$ & Resid. ($\\sigma$) \\\\")
    lines.append("\\midrule")
    for b in J["bins"]:
        resid = (b["A_lit"] - J["scale_model"])/b["sigma_lit"]
        lines.append(f"{latex_escape(b['survey'])} & {latex_escape(str(b['bin_id']))} & {fmt(b['z_eff'],2)} & "
                     f"{fmt(b['A_lit'])}$\\pm${fmt(b['sigma_lit'])} & {resid:+.2f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Phase-4 $g\\times\\kappa$ (multi-bin) per-bin details.}")
    lines.append("\\end{table}")
    pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    open(f"{OUTDIR}/gk_multi_table.tex","w").write("\n".join(lines)+"\n")
    print(f"Wrote {OUTDIR}/gk_multi_table.tex")

def main():
    write_single()
    write_multi()

if __name__ == "__main__":
    main()
