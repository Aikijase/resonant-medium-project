#!/usr/bin/env python3
"""
Full report (Phases 2 → 6)
Output → outputs/report_full.html
"""
import os, json, html, datetime
from pathlib import Path

READ = lambda p: open(p,"r",encoding="utf-8").read() if os.path.exists(p) else ""
EXISTS = os.path.exists

OUT = "outputs/report_full.html"
P2_MD  = "outputs/phase2/REPORT.md"
P4P1   = "plots/gk_residuals.png"
P4P2   = "plots/gk_multi_residuals.png"
P5P1   = "plots/phase5/phase5_stability_map.png"
P5P2   = "plots/phase5/phase5_damping_evolution.png"
P6P    = "plots/phase6/resonant_response.png"
P6JSON = "outputs/phase6/resonant_response.json"

def md2html(md:str)->str:
    out, in_code = [], False
    for ln in md.splitlines():
        if ln.startswith("```"):
            in_code = not in_code
            out.append("<pre><code>" if in_code else "</code></pre>")
        elif in_code:
            out.append(html.escape(ln))
        elif ln.startswith("#"):
            level = ln.count("#",0,ln.find(" ")); txt = ln.strip("# ").strip()
            out.append(f"<h{level}>{html.escape(txt)}</h{level}>")
        elif ln.strip().startswith("- "):
            if not(out and out[-1].startswith("<ul")): out.append("<ul>")
            out.append(f"<li>{html.escape(ln.strip()[2:])}</li>")
        elif not ln.strip():
            if out and out[-1].startswith("<li>"): out.append("</ul>")
            out.append("<p></p>")
        else:
            out.append(f"<p>{html.escape(ln)}</p>")
    if out and out[-1].startswith("<li>"): out.append("</ul>")
    return "\n".join(out)

def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    Path("outputs").mkdir(exist_ok=True)

    J6 = json.loads(READ(P6JSON)) if EXISTS(P6JSON) else {}
    stats = J6.get("stats_trimmed_10_90", J6.get("stats_finite", {}))
    s6 = f"""
    <h2>Phase 6 – Resonant Response</h2>
    <div class='plots'><img src='../{P6P}' alt='R(z)'></div>
    <p class='note'>A_ratio={J6.get('A_ratio','—')}, mean={stats.get('mean','—')}, std={stats.get('std','—')}, n={stats.get('n','—')}, z-window={J6.get('z_window','—')}</p>
    """

    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Resonant Medium — Full Report</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:24px;color:#111}}
.plots{{display:flex;gap:16px;flex-wrap:wrap}}
.plots img{{max-width:48%;min-width:320px;border:1px solid #eee;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.note{{font-size:.95em;color:#333}}
pre{{background:#f6f8fa;padding:10px;overflow-x:auto}}
</style></head><body>
<h1>Resonant Medium — Phases 2–6</h1>
<p class='note'>Generated {html.escape(now)}.</p>

<h2>Phase 4 Plots</h2>
<div class='plots'>
{'<img src=../'+P4P1+'> ' if EXISTS(P4P1) else ''}
{'<img src=../'+P4P2+'> ' if EXISTS(P4P2) else ''}
</div>

<h2>Phase 5 Plots</h2>
<div class='plots'>
{'<img src=../'+P5P1+'> ' if EXISTS(P5P1) else ''}
{'<img src=../'+P5P2+'> ' if EXISTS(P5P2) else ''}
</div>

{s6}

<h2>Phase 2 (verbatim)</h2>
<div class='md'>{md2html(READ(P2_MD))}</div>

</body></html>"""
    open(OUT,"w",encoding="utf-8").write(html_text)
    print(f"Wrote {OUT}")

if __name__ == "__main__": main()
