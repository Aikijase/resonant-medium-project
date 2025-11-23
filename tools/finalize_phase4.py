#!/usr/bin/env python3
import json, pathlib

OUT = pathlib.Path("outputs/phase4/lensing_score.json")
REP = pathlib.Path("outputs/phase2/REPORT.md")

def fmt(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except: return str(x)

def main():
    if not OUT.exists():
        print("No outputs/phase4/lensing_score.json found")
        return
    J = json.loads(OUT.read_text())

    lines = [
      "## Phase-4: CMB Lensing Cross-Check",
      f"- Combined χ²: **{fmt(J['chi2'])}** (n={J['n']}, k={J['k']})",
      f"- AIC: **{fmt(J['AIC']['value'])}**, ΔAIC vs LCDM(A_L=1): **{fmt(J['AIC']['delta'])}**",
      f"- WWI: **{J.get('WWI')}**",
      "",
      "| survey | A_model | A_obs ± σ | resid |",
      "|---|---:|---:|---:|",
    ]
    for r in J["rows"]:
        lines.append(f"| {r['survey']} | {fmt(r['A_model'])} | {fmt(r['A_L'])} ± {fmt(r['sigma_A'])} | {fmt(r['resid'])} |")

    if pathlib.Path("plots/cmb_lensing_amp.png").exists():
        lines.append("\n- Figure: `plots/cmb_lensing_amp.png`")

    REP.parent.mkdir(parents=True, exist_ok=True)
    if not REP.exists():
        REP.write_text("# Phase-2 Report\n")
    REP.write_text(REP.read_text() + "\n" + "\n".join(lines) + "\n")
    print(f"Updated {REP}")

if __name__ == "__main__":
    main()
