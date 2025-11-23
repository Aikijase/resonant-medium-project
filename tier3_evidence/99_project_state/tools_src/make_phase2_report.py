import json, pathlib, datetime
out = pathlib.Path("outputs/phase2")
fs8  = json.load(open(out/"fs8_eval.json"))
isw  = json.load(open(out/"isw_eval.json"))
join = json.load(open(out/"pred_eval.phase2_joint.json"))

ts = datetime.datetime.now().isoformat(timespec="seconds")
md = []
md += [f"# Phase-2 Mini-Report", "", f"_Generated: {ts}_", ""]
md += ["## Growth (fσ8)", 
       f"- χ²: **{fs8['chi2']:.3f}**",
       f"- AIC: **{fs8['AIC']['value']:.3f}**",
       f"- BIC: **{fs8['BIC']['value']:.3f}**",
       f"- WWI: **{fs8.get('WWI')}**",
       f"- Overlay plot: `plots/fs8_overlay.png`",
       f"- Residuals plot: `plots/fs8_residuals.png`", ""]
md += ["## ISW (placeholder)",
       f"- Bands: **{sum(len(b['ell']) for b in isw['bandpowers'])} ℓ-points**",
       "- Note: placeholder spectra; scoring disabled.", ""]
md += ["## Joint Summary",
       f"- File: `outputs/phase2/pred_eval.phase2_joint.json`",
       f"- phase1_WWI: **{join.get('phase1_WWI')}**",
       f"- fs8_WWI: **{join.get('fs8_WWI')}**",
       f"- isw_info: {join.get('isw_info')}", ""]
(open(out/"REPORT.md","w")).write("\n".join(md))
print("Wrote", out/"REPORT.md")
