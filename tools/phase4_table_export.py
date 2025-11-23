#!/usr/bin/env python3
"""
Export a compact Markdown table for g×κ to outputs/phase4/gk_table.md
Useful for quick notes/posts.

Reads outputs/phase4/gk_score.json (created by tools/phase4_gk_score.py).
"""
import json, os, math, pathlib

INP = "outputs/phase4/gk_score.json"
OUT = "outputs/phase4/gk_table.md"

def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def main():
    J = json.load(open(INP))
    D = J.get("details", [])
    rows = [
        "| Catalog | z_eff | A_lit ± σ | Residual (σ) |",
        "|---|---:|---:|---:|",
    ]
    for d in D:
        rows.append(f"| {d['name']} | {fmt(d['z_eff'],2)} | {fmt(d['A_lit'])} ± {fmt(d['sigma_lit'])} | {d['resid_sigma_model']:+.2f} |")

    rows.append("")
    rows.append(f"**Â** = {fmt(J.get('A_hat'))} ± {fmt(J.get('sigma_A_hat'))}   |   **A_pred** = {fmt(J.get('scale_model'))}   |   **pull** = { (J['scale_model']-J['A_hat'])/J['sigma_A_hat']:+.2f}σ")
    rows.append(f"χ²={fmt(J.get('chi2'))}, n={J.get('n')}, ⟨|res|⟩≈{math.sqrt(J['chi2']/max(J['n'],1)):.2f}σ, ΔAIC={fmt(J['AIC']['delta'])}, WWI={J.get('WWI')}")

    pathlib.Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)
    open(OUT, "w").write("\n".join(rows))
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
