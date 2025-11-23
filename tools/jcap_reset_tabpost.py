#!/usr/bin/env python3
import re, pathlib, json, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
paper = ROOT / "paper-jcap" / "main.tex"
fit   = ROOT / "outputs/phase20" / "p20_fit.json"

vals = {
    "tau":   {"mean": 0.300, "sd": 0.040, "units": "dimensionless", "sym": r"$\tauMem$"},
    "kappa": {"mean": -0.020,"sd": 0.010, "units": "dimensionless", "sym": r"$\kappaMem$"},
    "omega0":{"mean": 1.60,  "sd": 0.10,  "units": "rad/e-fold",   "sym": r"$\wzero$"},
    "Q":     {"mean": 0.830, "sd": 0.050, "units": "dimensionless", "sym": r"$\Qfact$"},
    "C_cal": {"mean": 1.25,  "sd": 0.10,  "units": r"(amplitude per $\kappa$)", "sym": r"$C_{\mathrm{cal}}$"},
}

def fm(m, s):
    def fmt(x, sd):
        if sd >= 0.1: return f"{x:.2f}"
        if sd >= 0.01: return f"{x:.3f}"
        return f"{x:.4f}"
    return f"${fmt(m,s)} \\pm {fmt(s,s)}$"

# Load fit JSON if present
if fit.exists():
    try:
        d = json.loads(fit.read_text(encoding="utf-8"))
        vals["tau"]["mean"],   vals["tau"]["sd"]   = d["tau"]["mean"],   d["tau"]["sd"]
        vals["kappa"]["mean"], vals["kappa"]["sd"] = d["kappa"]["mean"], d["kappa"]["sd"]
        vals["omega0"]["mean"],vals["omega0"]["sd"]= d["omega0"]["mean"],d["omega0"]["sd"]
        vals["Q"]["mean"],     vals["Q"]["sd"]     = d["Q"]["mean"],     d["Q"]["sd"]
        vals["C_cal"]["mean"], vals["C_cal"]["sd"] = d["C_cal"]["mean"], d["C_cal"]["sd"]
    except Exception as e:
        print(f"NOTE: could not parse {fit}: {e}; using defaults")

tex = paper.read_text(encoding="utf-8")

rows = [
    rf"{vals['tau']['sym']} & {fm(vals['tau']['mean'], vals['tau']['sd'])} & {vals['tau']['units']} \\",
    rf"{vals['kappa']['sym']} & {fm(vals['kappa']['mean'], vals['kappa']['sd'])} & {vals['kappa']['units']} \\",
    rf"{vals['omega0']['sym']} & {fm(vals['omega0']['mean'], vals['omega0']['sd'])} & {vals['omega0']['units']} \\",
    rf"{vals['Q']['sym']} & {fm(vals['Q']['mean'], vals['Q']['sd'])} & {vals['Q']['units']} \\",
    rf"{vals['C_cal']['sym']} & {fm(vals['C_cal']['mean'], vals['C_cal']['sd'])} & {vals['C_cal']['units']} \\",
]
clean_tab = (
    r"\begin{tabular}{lcc}" + "\n"
    r"\toprule" + "\n"
    r"Parameter & $Mean \pm SD$ & Units \\" + "\n"
    r"\midrule" + "\n" +
    "\n".join(rows) + "\n" +
    r"\bottomrule" + "\n" +
    r"\end{tabular}"
)

pattern = re.compile(
    r"(\\label\{tab:post\}.*?\n)(\\begin\{tabular\}\{lcc\}.*?\\end\{tabular\})",
    re.S
)
if not pattern.search(tex):
    print("ERROR: Could not locate the tabular block for \\label{tab:post}.")
    sys.exit(1)

tex_fixed = pattern.sub(r"\1" + clean_tab, tex, count=1)
tex_fixed = tex_fixed.replace("-e.300", "0.300")
tex_fixed = re.sub(r"(\$[0-9\.\-]+\s*\\pm\s*[0-9\.]+\$)[0-9\.\s]*\\pm\s*[0-9\.]+", r"\1", tex_fixed)

paper.write_text(tex_fixed, encoding="utf-8")
print("tab:post table block replaced cleanly.")
