#!/usr/bin/env python3
"""
Generate a styled HTML report from the Thrace best bundle.

Inputs (defaults to outputs/thrace_best/):
  - joint.std_summary.json  (LCDM/RESN + deltas)
  - params.json             (alpha, gamma, beta_de, beta_dm)
  - joint.metrics.png       (optional)
  - recycling.solution.csv  (linked)

Output:
  - joint.report.html

Usage:
  PYTHONPATH=. python tools/thrace_make_html_report.py \
    --best-dir outputs/thrace_best
"""
from __future__ import annotations
import argparse, json, os, datetime as dt
from pathlib import Path
import html

ROOT = Path(__file__).resolve().parents[1]

def fmt(x, nd=6):
    if x is None: return "—"
    try:
        xv = float(x)
        return f"{xv:.{nd}f}" if abs(xv) < 1e6 else f"{xv:.{nd}e}"
    except Exception:
        return html.escape(str(x))

def row(label, value):
    return f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>"

def main():
    ap = argparse.ArgumentParser(description="Build a standalone HTML report for Thrace best bundle.")
    ap.add_argument("--best-dir", default="outputs/thrace_best", help="Directory containing best artifacts")
    args = ap.parse_args()

    best = ROOT / args.best_dir
    if not best.exists():
        raise SystemExit(f"[report-html] not found: {best}")

    # Load JSONs
    std_path = best / "joint.std_summary.json"
    prm_path = best / "params.json"
    std = json.loads(std_path.read_text()) if std_path.exists() else {}
    prm = json.loads(prm_path.read_text()) if prm_path.exists() else {}

    # Optional assets
    metrics_png = "joint.metrics.png" if (best / "joint.metrics.png").exists() else None
    recycling_csv = "recycling.solution.csv" if (best / "recycling.solution.csv").exists() else None

    # Pull values
    chi2 = std.get("chi2"); AIC = std.get("AIC"); BIC = std.get("BIC")
    dChi2 = std.get("dChi2"); dAIC = std.get("dAIC"); dBIC = std.get("dBIC")
    models = std.get("models", {}) or {}
    lcdm = models.get("lcdm", {}) or {}
    resn = models.get("resn", {}) or {}

    alpha = prm.get("alpha"); gamma = prm.get("gamma")
    beta_de = prm.get("beta_de"); beta_dm = prm.get("beta_dm")

    # HTML skeleton
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Helvetica Neue',Arial,sans-serif;margin:32px;background:#0b0d11;color:#e6e8ec}
    h1,h2,h3{color:#fff;margin:0 0 12px}
    .card{background:#12151b;border:1px solid #1f2430;border-radius:14px;padding:16px;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,.25)}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px 10px;border-bottom:1px solid #1f2430;text-align:left}
    th{color:#9aa4b2;font-weight:600;width:220px}
    code,kbd{background:#0f131a;border:1px solid #1b222e;border-radius:8px;padding:2px 6px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
    .muted{color:#9aa4b2}
    img{max-width:100%;border-radius:10px;border:1px solid #1f2430}
    a{color:#7ab7ff;text-decoration:none} a:hover{text-decoration:underline}
    .pill{display:inline-block;background:#162033;border:1px solid #24314a;color:#b8c7e6;border-radius:999px;padding:4px 10px;margin-right:8px}
    """
    html_head = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Thrace Best Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{css}</style></head><body>
<h1>Thrace — Best Run Summary</h1>
<p class="muted">Generated {html.escape(now)}</p>
"""

    # Topline
    topline = f"""
<div class="card">
  <h2>Topline</h2>
  <div class="grid">
    <div class="card">
      <h3>Canonical Metrics</h3>
      <table>
        {row("χ²", fmt(chi2))}
        {row("AIC", fmt(AIC))}
        {row("BIC", fmt(BIC))}
        {row("Δχ² (LCDM−RESN)", fmt(dChi2))}
        {row("ΔAIC (LCDM−RESN)", fmt(dAIC))}
        {row("ΔBIC (LCDM−RESN)", fmt(dBIC))}
      </table>
    </div>
    <div class="card">
      <h3>Parameters</h3>
      <table>
        {row("α (alpha)", fmt(alpha))}
        {row("γ (gamma)", fmt(gamma))}
        {row("β_de", fmt(beta_de))}
        {row("β_dm", fmt(beta_dm))}
      </table>
    </div>
  </div>
</div>
"""

    # Models
    def model_table(title, m):
        keys = ["Om","A","f","phi","M","beta","chi2","AIC","BIC","k","N"]
        rows = "\n".join(row(k, fmt(m.get(k))) for k in keys if k in m)
        return f'<div class="card"><h3>{title}</h3><table>{rows}</table></div>'

    models_html = f"""
<div class="card">
  <h2>Models</h2>
  <div class="grid">
    {model_table("LCDM", lcdm) if lcdm else ""}
    {model_table("RESN (canonical)", resn) if resn else ""}
  </div>
</div>
"""

    # Assets
    assets = '<div class="card"><h2>Artifacts</h2><div class="grid">'
    if metrics_png:
        assets += f'<div class="card"><h3>Metrics</h3><img src="{html.escape(metrics_png)}" alt="metrics"></div>'
    if recycling_csv:
        assets += f'<div class="card"><h3>Recycling Solution</h3><p><a href="{html.escape(recycling_csv)}">Download recycling.solution.csv</a></p></div>'
    assets += "</div></div>"

    # Footer
    footer = "<p class='muted'>Provenance: generated from outputs/thrace_best/.</p></body></html>"

    out_path = best / "joint.report.html"
    out_path.write_text(html_head + topline + models_html + assets + footer)
    print(f"[report-html] wrote {out_path}")

if __name__ == "__main__":
    main()
