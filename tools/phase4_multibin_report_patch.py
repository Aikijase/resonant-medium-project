#!/usr/bin/env python3
"""
Insert/update a '## Phase-4: g×κ (multi-bin)' section in outputs/phase2/REPORT.md
using outputs/phase4/gk_multi.json.
"""
import json, os, re, math, pathlib

REPORT = "outputs/phase2/REPORT.md"
JSONIN = "outputs/phase4/gk_multi.json"

def fmt(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except: return "—"

def build_section(J):
    n = J["n"]
    mean_sigma = math.sqrt(J["chi2_pred"]/max(n,1))
    pull = (J["scale_model"] - J["A_hat"]) / J["sigma_A_hat"]
    lines = []
    lines.append("## Phase-4: g×κ (multi-bin)")
    lines.append("")
    lines.append(f"- Overall: n={n}, ⟨|res|⟩≈{mean_sigma:.2f}σ, ΔAIC={fmt(J['AIC']['delta'])}, WWI={J['WWI']}")
    lines.append(f"- Â={fmt(J['A_hat'])} ± {fmt(J['sigma_A_hat'])} | A_pred={fmt(J['scale_model'])} | pull={pull:+.2f}σ")
    lines.append("")
    lines += ["### Per-survey summary", "", "| Survey | n | Â ± σ | ⟨|res|⟩ (σ) | ΔAIC | WWI |", "|---|---:|---:|---:|---:|---:|"]
    for sname, S in sorted(J["surveys"].items()):
        ms = math.sqrt(S["chi2_pred"]/max(S["n"],1))
        lines.append(f"| {sname} | {S['n']} | {fmt(S['A_hat'])} ± {fmt(S['sigma_A_hat'])} | {ms:.2f} | {fmt(S['AIC']['delta'])} | {S['WWI']} |")
    lines.append("")
    lines += ["### Per-bin details", "", "| Survey | Bin | z_eff | A_lit ± σ | Resid (σ) |", "|---|---:|---:|---:|---:|"]
    for b in J["bins"]:
        resid = (b["A_lit"] - J["scale_model"]) / b["sigma_lit"]
        lines.append(f"| {b['survey']} | {b['bin_id']} | {fmt(b['z_eff'],2)} | {fmt(b['A_lit'])} ± {fmt(b['sigma_lit'])} | {resid:+.2f} |")
    lines.append("")
    return "\n".join(lines)

def upsert(report_text, section_text):
    pat = re.compile(r"^## Phase-4: g×κ \(multi-bin\).*?(?=^## |\Z)", re.DOTALL|re.MULTILINE)
    if pat.search(report_text):
        return pat.sub(section_text, report_text)
    if not report_text.endswith("\n"): report_text += "\n"
    return report_text + "\n" + section_text + "\n"

def main():
    if not os.path.exists(JSONIN):
        print("[warn] no multi-bin JSON to patch"); return
    J = json.load(open(JSONIN))
    pathlib.Path("outputs/phase2").mkdir(parents=True, exist_ok=True)
    text = open(REPORT).read() if os.path.exists(REPORT) else "# Phase-2 Report\n\n"
    sec = build_section(J)
    out = upsert(text, sec)
    open(REPORT, "w").write(out)
    print(f"Updated {REPORT} (multi-bin)")

if __name__ == "__main__":
    main()
