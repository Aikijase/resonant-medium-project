#!/usr/bin/env python3
"""
Build a self-contained HTML report for Phase-4 (and embed Phase-2 blurb).

Inputs:
  - outputs/phase2/REPORT.md            (if present; appended as text block)
  - outputs/phase4/gk_score.json
  - outputs/phase4/gk_multi.json
  - plots/gk_residuals.png
  - plots/gk_multi_residuals.png

Output:
  - outputs/phase4/phase4_report.html

No external dependencies. Keeps paths relative so you can zip & share.
"""
import os, json, math, html, datetime
from pathlib import Path

P2_MD     = "outputs/phase2/REPORT.md"
GK_JSON   = "outputs/phase4/gk_score.json"
MB_JSON   = "outputs/phase4/gk_multi.json"
PLOT1     = "plots/gk_residuals.png"
PLOT2     = "plots/gk_multi_residuals.png"
OUT_HTML  = "outputs/phase4/phase4_report.html"

def fmt(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except: return "—"

def read(path):
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read()
    return None

def md_to_html(md_text):
    # ultra-minimal markdown to html (headers + code + paragraphs)
    # (good enough for this report; avoids external tools)
    lines = (md_text or "").splitlines()
    out = []
    in_code = False
    for ln in lines:
        if ln.startswith("```"):
            in_code = not in_code
            out.append("<pre><code>" if in_code else "</code></pre>")
            continue
        if in_code:
            out.append(html.escape(ln))
            continue
        if ln.startswith("### "):
            out.append(f"<h3>{html.escape(ln[4:])}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{html.escape(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{html.escape(ln[2:])}</h1>")
        elif ln.strip().startswith("- "):
            # simple unordered list detection
            if not (out and out[-1].startswith("<ul")):
                out.append("<ul>")
            out.append(f"<li>{html.escape(ln.strip()[2:])}</li>")
            # close UL when next non-list line arrives; handled below
        elif ln.strip()=="":
            # close list if needed
            if out and out[-1].startswith("<li>"):
                out.append("</ul>")
            out.append("<p></p>")
        else:
            out.append(f"<p>{html.escape(ln)}</p>")
    if out and out[-1].startswith("<li>"):
        out.append("</ul>")
    return "\n".join(out)

def table_from_gk_single(J):
    rows = J.get("details", [])
    html_rows = "\n".join(
        f"<tr><td>{html.escape(r['name'])}</td>"
        f"<td style='text-align:right'>{fmt(r['z_eff'],2)}</td>"
        f"<td style='text-align:right'>{fmt(r['A_lit'])} ± {fmt(r['sigma_lit'])}</td>"
        f"<td style='text-align:right'>{r['resid_sigma_model']:+.2f}</td></tr>"
        for r in rows
    )
    summary = (f"Â = {fmt(J.get('A_hat'))} ± {fmt(J.get('sigma_A_hat'))} | "
               f"A_pred = {fmt(J.get('scale_model'))} | "
               f"⟨|res|⟩ ≈ {math.sqrt(J['chi2']/max(J['n'],1)):.2f}σ | "
               f"ΔAIC = {fmt(J['AIC']['delta'])} | WWI = {J.get('WWI')}")
    return f"""
    <h3>Single-catalog g×κ</h3>
    <table class='nice'>
      <thead><tr><th>Catalog</th><th>z_eff</th><th>A_lit ± σ</th><th>Residual (σ)</th></tr></thead>
      <tbody>
        {html_rows}
      </tbody>
    </table>
    <p class='note'>{html.escape(summary)}</p>
    """

def table_from_gk_multi(J):
    # per-survey summary
    survey_rows = "\n".join(
        f"<tr><td>{html.escape(sname)}</td>"
        f"<td style='text-align:right'>{S['n']}</td>"
        f"<td style='text-align:right'>{fmt(S['A_hat'])} ± {fmt(S['sigma_A_hat'])}</td>"
        f"<td style='text-align:right'>{math.sqrt(S['chi2_pred']/max(S['n'],1)):.2f}</td>"
        f"<td style='text-align:right'>{fmt(S['AIC']['delta'])}</td>"
        f"<td style='text-align:right'>{S['WWI']}</td></tr>"
        for sname,S in sorted(J["surveys"].items())
    )
    # per-bin table
    bin_rows = "\n".join(
        f"<tr><td>{html.escape(b['survey'])}</td>"
        f"<td style='text-align:right'>{html.escape(str(b['bin_id']))}</td>"
        f"<td style='text-align:right'>{fmt(b['z_eff'],2)}</td>"
        f"<td style='text-align:right'>{fmt(b['A_lit'])} ± {fmt(b['sigma_lit'])}</td>"
        f"<td style='text-align:right'>{(b['A_lit']-J['scale_model'])/b['sigma_lit']:+.2f}</td></tr>"
        for b in J["bins"]
    )
    overall = (f"Overall: n={J['n']} | ⟨|res|⟩ ≈ {math.sqrt(J['chi2_pred']/max(J['n'],1)):.2f}σ | "
               f"ΔAIC = {fmt(J['AIC']['delta'])} | WWI = {J['WWI']} | "
               f"Â = {fmt(J['A_hat'])} ± {fmt(J['sigma_A_hat'])} | "
               f"A_pred = {fmt(J['scale_model'])}")
    return f"""
    <h3>Multi-bin g×κ</h3>
    <p class='note'>{html.escape(overall)}</p>
    <h4>Per-survey summary</h4>
    <table class='nice'>
      <thead><tr><th>Survey</th><th>n</th><th>Â ± σ</th><th>⟨|res|⟩ (σ)</th><th>ΔAIC</th><th>WWI</th></tr></thead>
      <tbody>
        {survey_rows}
      </tbody>
    </table>
    <h4>Per-bin details</h4>
    <table class='nice'>
      <thead><tr><th>Survey</th><th>Bin</th><th>z_eff</th><th>A_lit ± σ</th><th>Residual (σ)</th></tr></thead>
      <tbody>
        {bin_rows}
      </tbody>
    </table>
    """

def main():
    Path("outputs/phase4").mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")

    # load JSONs
    J_single = json.load(open(GK_JSON)) if os.path.exists(GK_JSON) else None
    J_multi  = json.load(open(MB_JSON)) if os.path.exists(MB_JSON) else None
    p2_md    = read(P2_MD)

    # sections
    s_single = table_from_gk_single(J_single) if J_single else "<p>No single-catalog JSON found.</p>"
    s_multi  = table_from_gk_multi(J_multi) if J_multi else "<p>No multi-bin JSON found.</p>"
    s_p2     = f"<h2>Phase-2 Report (verbatim)</h2>\n<div class='md'>{md_to_html(p2_md)}</div>" if p2_md else ""

    # plots
    img1 = f"<img src='../{PLOT1}' alt='gk residuals'/>" if os.path.exists(PLOT1) else ""
    img2 = f"<img src='../{PLOT2}' alt='gk multi residuals'/>" if os.path.exists(PLOT2) else ""

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Phase-4 Lensing Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}
  h1, h2, h3 {{ margin: 0.6em 0 0.3em; }}
  .note {{ color:#333; font-size: 0.95em; }}
  table.nice {{ border-collapse: collapse; margin: 12px 0; width: 100%; }}
  table.nice th, table.nice td {{ border: 1px solid #ddd; padding: 6px 8px; }}
  table.nice th {{ background:#fafafa; text-align:left; }}
  .plots {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .plots img {{ max-width: 48%; min-width: 320px; border:1px solid #eee; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  pre {{ background:#f6f8fa; padding:10px; overflow-x:auto; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em; }}
</style>
</head>
<body>
  <h1>Phase-4: CMB (κκ) & Galaxy×κ Lensing</h1>
  <p class='note'>Generated {html.escape(now)}. This document summarizes single-catalog and multi-bin g×κ fits with growth-driven predictions.</p>

  <div class="plots">
    {img1}
    {img2}
  </div>

  {s_single}
  {s_multi}
  {s_p2}
</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"Wrote {OUT_HTML}")

if __name__ == "__main__":
    main()
