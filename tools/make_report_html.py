#!/usr/bin/env python3
"""
Make a single self-contained HTML report:
- Phase-2 REPORT.md (verbatim section)
- Phase-4 g×κ single + multi-bin tables and residual plots
- Phase-5 stability map + damping ODE plots + summary

Outputs: outputs/report_full.html
No external deps.
"""
import os, json, math, html, datetime
from pathlib import Path

# Inputs
P2_MD   = "outputs/phase2/REPORT.md"
GK_JSON = "outputs/phase4/gk_score.json"
MB_JSON = "outputs/phase4/gk_multi.json"
P4P1    = "plots/gk_residuals.png"
P4P2    = "plots/gk_multi_residuals.png"

P5_GRID = "outputs/phase5/stability_grid.json"
P5_SUM  = "outputs/phase5/phase5_summary.md"
P5P1    = "plots/phase5/phase5_stability_map.png"
P5P2    = "plots/phase5/phase5_damping_evolution.png"

OUT_HTML = "outputs/report_full.html"

def fmt(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except: return "—"

def read(path):
    return open(path,"r",encoding="utf-8").read() if os.path.exists(path) else None

def md_to_html(md_text):
    lines = (md_text or "").splitlines()
    out=[]; in_code=False
    for ln in lines:
        if ln.startswith("```"):
            in_code = not in_code
            out.append("<pre><code>" if in_code else "</code></pre>")
            continue
        if in_code:
            out.append(html.escape(ln))
            continue
        if ln.startswith("### "): out.append(f"<h3>{html.escape(ln[4:])}</h3>")
        elif ln.startswith("## "): out.append(f"<h2>{html.escape(ln[3:])}</h2>")
        elif ln.startswith("# "): out.append(f"<h1>{html.escape(ln[2:])}</h1>")
        elif ln.strip().startswith("- "):
            if not (out and out[-1].startswith("<ul")): out.append("<ul>")
            out.append(f"<li>{html.escape(ln.strip()[2:])}</li>")
        elif ln.strip()=="":
            if out and out[-1].startswith("<li>"): out.append("</ul>")
            out.append("<p></p>")
        else:
            out.append(f"<p>{html.escape(ln)}</p>")
    if out and out[-1].startswith("<li>"): out.append("</ul>")
    return "\n".join(out)

def section_phase4_single(J):
    if not J: return "<p>No Phase-4 single-catalog JSON found.</p>"
    rows = J.get("details", [])
    trs = "\n".join(
        f"<tr><td>{html.escape(r['name'])}</td>"
        f"<td class='num'>{fmt(r['z_eff'],2)}</td>"
        f"<td class='num'>{fmt(r['A_lit'])} ± {fmt(r['sigma_lit'])}</td>"
        f"<td class='num'>{r['resid_sigma_model']:+.2f}</td></tr>"
        for r in rows
    )
    summary = (f"Â={fmt(J.get('A_hat'))}±{fmt(J.get('sigma_A_hat'))}  |  "
               f"A_pred={fmt(J.get('scale_model'))}  |  "
               f"⟨|res|⟩≈{math.sqrt(J['chi2']/max(J['n'],1)):.2f}σ  |  "
               f"ΔAIC={fmt(J['AIC']['delta'])}  |  WWI={J.get('WWI')}")
    return f"""
    <h3>Phase-4: g×κ (single)</h3>
    <table class='nice'>
      <thead><tr><th>Catalog</th><th>z_eff</th><th>A_lit ± σ</th><th>Residual (σ)</th></tr></thead>
      <tbody>{trs}</tbody>
    </table>
    <p class='note'>{html.escape(summary)}</p>
    """

def section_phase4_multi(J):
    if not J: return "<p>No Phase-4 multi-bin JSON found.</p>"
    srows = "\n".join(
        f"<tr><td>{html.escape(sname)}</td>"
        f"<td class='num'>{S['n']}</td>"
        f"<td class='num'>{fmt(S['A_hat'])} ± {fmt(S['sigma_A_hat'])}</td>"
        f"<td class='num'>{math.sqrt(S['chi2_pred']/max(S['n'],1)):.2f}</td>"
        f"<td class='num'>{fmt(S['AIC']['delta'])}</td>"
        f"<td class='num'>{S['WWI']}</td></tr>"
        for sname,S in sorted(J["surveys"].items())
    )
    brows = "\n".join(
        f"<tr><td>{html.escape(b['survey'])}</td>"
        f"<td class='num'>{html.escape(str(b['bin_id']))}</td>"
        f"<td class='num'>{fmt(b['z_eff'],2)}</td>"
        f"<td class='num'>{fmt(b['A_lit'])} ± {fmt(b['sigma_lit'])}</td>"
        f"<td class='num'>{(b['A_lit']-J['scale_model'])/b['sigma_lit']:+.2f}</td></tr>"
        for b in J["bins"]
    )
    overall = (f"Overall: n={J['n']}  |  ⟨|res|⟩≈{math.sqrt(J['chi2_pred']/max(J['n'],1)):.2f}σ  |  "
               f"ΔAIC={fmt(J['AIC']['delta'])}  |  WWI={J['WWI']}  |  "
               f"Â={fmt(J['A_hat'])}±{fmt(J['sigma_A_hat'])}  |  "
               f"A_pred={fmt(J['scale_model'])}")
    return f"""
    <h3>Phase-4: g×κ (multi-bin)</h3>
    <p class='note'>{html.escape(overall)}</p>
    <h4>Per-survey</h4>
    <table class='nice'>
      <thead><tr><th>Survey</th><th>n</th><th>Â ± σ</th><th>⟨|res|⟩ (σ)</th><th>ΔAIC</th><th>WWI</th></tr></thead>
      <tbody>{srows}</tbody>
    </table>
    <h4>Per-bin details</h4>
    <table class='nice'>
      <thead><tr><th>Survey</th><th>Bin</th><th>z_eff</th><th>A_lit ± σ</th><th>Residual (σ)</th></tr></thead>
      <tbody>{brows}</tbody>
    </table>
    """

def section_phase5():
    grid = read(P5_GRID)
    summ = read(P5_SUM)
    p = []
    p.append("<h2>Phase-5: Stability & Stress-Test</h2>")
    if os.path.exists(P5P1) or os.path.exists(P5P2):
        p.append("<div class='plots'>")
        if os.path.exists(P5P1):
            p.append(f"<img src='{html.escape('../'+P5P1)}' alt='stability map'/>")
        if os.path.exists(P5P2):
            p.append(f"<img src='{html.escape('../'+P5P2)}' alt='damping evolution'/>")
        p.append("</div>")
    if summ:
        p.append("<h3>Summary</h3>")
        p.append("<div class='md'>"+md_to_html(summ)+"</div>")
    else:
        p.append("<p>No Phase-5 summary found.</p>")
    return "\n".join(p)

def main():
    Path("outputs").mkdir(exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")

    J_single = json.loads(read(GK_JSON)) if os.path.exists(GK_JSON) else None
    J_multi  = json.loads(read(MB_JSON)) if os.path.exists(MB_JSON) else None
    p2_md    = read(P2_MD)

    sec_p2   = f"<h2>Phase-2 (verbatim)</h2><div class='md'>{md_to_html(p2_md)}</div>" if p2_md else ""
    sec_p4s  = section_phase4_single(J_single)
    sec_p4m  = section_phase4_multi(J_multi)
    sec_p5   = section_phase5()

    html_text = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Resonant Medium — Phases 2, 4 & 5 Report</title>
<style>
 body {{ font-family: system-ui, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}
 h1,h2,h3,h4 {{ margin: .6em 0 .3em; }}
 .note {{ color:#333; font-size:.95em }}
 table.nice {{ border-collapse:collapse; width:100%; margin:12px 0 }}
 table.nice th, table.nice td {{ border:1px solid #ddd; padding:6px 8px }}
 table.nice th {{ background:#fafafa; text-align:left }}
 td.num {{ text-align:right }}
 .plots {{ display:flex; gap:16px; flex-wrap:wrap }}
 .plots img {{ max-width:48%; min-width:320px; border:1px solid #eee; box-shadow:0 1px 3px rgba(0,0,0,.05) }}
 pre {{ background:#f6f8fa; padding:10px; overflow-x:auto }}
 code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.9em }}
</style>
</head>
<body>
  <h1>Resonant Medium — Consolidated Report</h1>
  <p class='note'>Generated {html.escape(now)}. Includes Phase-2 overview, Phase-4 lensing results, and Phase-5 stability diagnostics.</p>

  <h2>Phase-4 Plots</h2>
  <div class="plots">
    {"<img src='../"+P4P1+"' alt='gk residuals'/>" if os.path.exists(P4P1) else ""}
    {"<img src='../"+P4P2+"' alt='gk multi residuals'/>" if os.path.exists(P4P2) else ""}
  </div>

  {sec_p4s}
  {sec_p4m}
  {sec_p5}
  {sec_p2}
</body></html>"""
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    open(OUT_HTML,"w",encoding="utf-8").write(html_text)
    print(f"Wrote {OUT_HTML}")

if __name__ == "__main__":
    main()
