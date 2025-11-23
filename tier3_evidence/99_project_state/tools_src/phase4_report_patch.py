#!/usr/bin/env python3
"""
Phase-4 report patcher:
- Reads outputs/phase4/lensing_score.json (κκ) and outputs/phase4/gk_score.json (g×κ)
- Creates or updates a '## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)' section
  in outputs/phase2/REPORT.md, including a compact table and bullet diagnostics.
- Idempotent: replaces the whole Phase-4 section if present, else appends.
"""
import json, os, re, pathlib, math

REPORT = "outputs/phase2/REPORT.md"
KK_JSON = "outputs/phase4/lensing_score.json"
GK_JSON = "outputs/phase4/gk_score.json"

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def build_section(Jkk, Jgk):
    lines = []
    lines.append("## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)")
    lines.append("")
    # κκ block
    if Jkk:
        nkk = int(Jkk.get("n", 1))
        chi2kk = float(Jkk.get("chi2", 0.0))
        mean_sigma_kk = math.sqrt(chi2kk/max(nkk,1))
        aic_kk = Jkk.get("AIC", {}).get("value", None)
        dAIC_kk = Jkk.get("AIC", {}).get("delta", None)
        wwi_kk = Jkk.get("WWI")
        lines += [
            "### κκ (CMB lensing auto)",
            f"- n={nkk}, χ²={fmt(chi2kk)}, ⟨|res|⟩≈{fmt(mean_sigma_kk,2)}σ",
            f"- AIC={fmt(aic_kk)}, ΔAIC={fmt(dAIC_kk)}, WWI={wwi_kk}",
            ""
        ]
    # g×κ block
    if Jgk:
        ngk = int(Jgk.get("n", 0))
        chi2gk = float(Jgk.get("chi2", 0.0))
        mean_sigma_gk = math.sqrt(chi2gk/max(ngk,1)) if ngk else float("nan")
        aic_gk = Jgk.get("AIC", {}).get("value", None)
        dAIC_gk = Jgk.get("AIC", {}).get("delta", None)
        wwi_gk = Jgk.get("WWI")
        Ahat = Jgk.get("A_hat"); sA = Jgk.get("sigma_A_hat"); Ap = Jgk.get("scale_model")
        dchi = Jgk.get("delta_chi2_pred")
        pull = (Ap - Ahat)/sA if (Ahat is not None and sA and Ap is not None) else None
        lines += [
            "### g×κ (galaxy–lensing, shape-only)",
            f"- n={ngk}, χ²={fmt(chi2gk)}, ⟨|res|⟩≈{fmt(mean_sigma_gk,2)}σ",
            f"- AIC={fmt(aic_gk)}, ΔAIC={fmt(dAIC_gk)}, WWI={wwi_gk}",
        ]
        if pull is not None:
            lines += [f"- Â={fmt(Ahat)} ± {fmt(sA)} | A_pred={fmt(Ap)} | pull={pull:+.2f}σ (Δχ²_pred={fmt(dchi)})"]
        lines.append("")
        # Table
        D = Jgk.get("details", [])
        if D:
            lines += [
                "| Catalog | z_eff | A_lit ± σ | Residual (σ) |",
                "|---|---:|---:|---:|",
            ]
            for d in D:
                lines.append(f"| {d['name']} | {fmt(d['z_eff'],2)} | {fmt(d['A_lit'])} ± {fmt(d['sigma_lit'])} | {d['resid_sigma_model']:+.2f} |")
            lines.append("")
    return "\n".join(lines) + "\n"

def upsert_section(report_text, new_section):
    # Replace from '## Phase-4:' to next '## ' or EOF
    pattern = re.compile(r"^## Phase-4:.*?(?=^## |\Z)", flags=re.DOTALL|re.MULTILINE)
    if pattern.search(report_text):
        return pattern.sub(new_section, report_text)
    else:
        if not report_text.endswith("\n"):
            report_text += "\n"
        return report_text + "\n" + new_section

def main():
    pathlib.Path("outputs/phase2").mkdir(parents=True, exist_ok=True)
    txt = ""
    if os.path.exists(REPORT):
        with open(REPORT, "r") as f:
            txt = f.read()
    else:
        txt = "# Phase-2 Report\n\n"

    Jkk = load_json(KK_JSON)
    Jgk = load_json(GK_JSON)
    new_sec = build_section(Jkk, Jgk)
    out_txt = upsert_section(txt, new_sec)

    with open(REPORT, "w") as f:
        f.write(out_txt)
    print(f"Updated {REPORT}")

if __name__ == "__main__":
    main()
