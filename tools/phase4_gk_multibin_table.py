#!/usr/bin/env python3
"""
Export Markdown tables from outputs/phase4/gk_multi.json:
 - Overall summary line
 - Per-survey summary table
 - Full per-bin table
Writes: outputs/phase4/gk_multi_table.md
"""
import json, math, os, pathlib

INP = "outputs/phase4/gk_multi.json"
OUT = "outputs/phase4/gk_multi_table.md"

def fmt(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except: return "—"

def main():
    J = json.load(open(INP))
    lines = []
    # overall
    n = J["n"]; Ahat = J["A_hat"]; sA = J["sigma_A_hat"]; Ap = J["scale_model"]
    pull = (Ap - Ahat)/sA
    mean_sigma = math.sqrt(J["chi2_pred"]/max(n,1))
    lines.append(f"**Overall**  Â = {fmt(Ahat)} ± {fmt(sA)}  |  A_pred = {fmt(Ap)}  |  pull = {pull:+.2f}σ  |  ⟨|res|⟩≈{mean_sigma:.2f}σ  |  ΔAIC={fmt(J['AIC']['delta'])}  WWI={J['WWI']}")
    lines.append("")
    # per-survey summary
    lines += ["### Per-survey summary", "", "| Survey | n | Â ± σ | ⟨|res|⟩ (σ) | ΔAIC | WWI |", "|---|---:|---:|---:|---:|---:|"]
    for sname, S in sorted(J["surveys"].items()):
        mean_s = math.sqrt(S["chi2_pred"]/max(S["n"],1))
        lines.append(f"| {sname} | {S['n']} | {fmt(S['A_hat'])} ± {fmt(S['sigma_A_hat'])} | {mean_s:.2f} | {fmt(S['AIC']['delta'])} | {S['WWI']} |")
    lines.append("")
    # per-bin table
    lines += ["### Per-bin details", "", "| Survey | Bin | z_eff | A_lit ± σ | Resid (σ) |", "|---|---:|---:|---:|---:|"]
    for b in J["bins"]:
        lines.append(f"| {b['survey']} | {b['bin_id']} | {fmt(b['z_eff'],2)} | {fmt(b['A_lit'])} ± {fmt(b['sigma_lit'])} | {b.get('resid_sigma_model','—')} |")
    pathlib.Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
