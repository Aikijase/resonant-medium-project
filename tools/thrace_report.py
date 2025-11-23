#!/usr/bin/env python3
"""
Pretty-print Thrace pipeline results from a std_summary.json and optionally
emit a small Markdown report.

Usage:
  PYTHONPATH=. python tools/thrace_report.py \
    --summary outputs/thrace_summaries/joint_baseline.std_summary.json \
    --md outputs/thrace_summaries/joint_baseline.report.md
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

def fmt(x, nd=6):
    if x is None: return "—"
    try:
        x = float(x)
        return f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.{nd}e}"
    except Exception:
        return str(x)

def row(k, v):
    print(f"{k:<10} : {v}")

def main():
    ap = argparse.ArgumentParser(description="Render Thrace std_summary.json nicely.")
    ap.add_argument("--summary", required=True, help="Path to *.std_summary.json")
    ap.add_argument("--md", default=None, help="Optional path to write a Markdown report")
    args = ap.parse_args()

    p = Path(args.summary)
    if not p.exists():
        sys.exit(f"[thrace-report] summary not found: {p}")

    data = json.loads(p.read_text())

    # console pretty-print
    print("=== Thrace summary ===")
    row("chi2", fmt(data.get("chi2")))
    row("AIC",  fmt(data.get("AIC")))
    row("BIC",  fmt(data.get("BIC")))
    if "dChi2" in data or "dAIC" in data or "dBIC" in data:
        print("--- deltas (LCDM−RESN) ---")
        if "dChi2" in data: row("Δχ²", fmt(data.get("dChi2")))
        if "dAIC"  in data: row("ΔAIC", fmt(data.get("dAIC")))
        if "dBIC"  in data: row("ΔBIC", fmt(data.get("dBIC")))

    models = data.get("models", {})
    lcdm   = models.get("lcdm", {})
    resn   = models.get("resn", {})
    if lcdm or resn:
        print("\n=== Models ===")
        if lcdm:
            print("LCDM:")
            for k in ["Om","M","beta","chi2","AIC","BIC","k","N"]:
                if k in lcdm:
                    row(f"  {k}", fmt(lcdm[k]))
        if resn:
            print("RESN:")
            for k in ["Om","A","f","phi","M","beta","chi2","AIC","BIC","k","N"]:
                if k in resn:
                    row(f"  {k}", fmt(resn[k]))

    # optional markdown report
    if args.md:
        mdp = Path(args.md)
        mdp.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Thrace Fit Report\n")
        lines.append("## Topline\n")
        lines.append(f"- **χ² (canonical)**: `{fmt(data.get('chi2'))}`")
        lines.append(f"- **AIC**: `{fmt(data.get('AIC'))}`")
        lines.append(f"- **BIC**: `{fmt(data.get('BIC'))}`\n")
        if "dChi2" in data or "dAIC" in data or "dBIC" in data:
            lines.append("## Deltas (LCDM − RESN)\n")
            if "dChi2" in data: lines.append(f"- Δχ²: `{fmt(data.get('dChi2'))}`")
            if "dAIC"  in data: lines.append(f"- ΔAIC: `{fmt(data.get('dAIC'))}`")
            if "dBIC"  in data: lines.append(f"- ΔBIC: `{fmt(data.get('dBIC'))}`")
            lines.append("")
        if lcdm or resn:
            lines.append("## Models\n")
            if lcdm:
                lines.append("### LCDM")
                for k in ["Om","M","beta","chi2","AIC","BIC","k","N"]:
                    if k in lcdm:
                        lines.append(f"- {k}: `{fmt(lcdm[k])}`")
                lines.append("")
            if resn:
                lines.append("### RESN")
                for k in ["Om","A","f","phi","M","beta","chi2","AIC","BIC","k","N"]:
                    if k in resn:
                        lines.append(f"- {k}: `{fmt(resn[k])}`")
                lines.append("")
        mdp.write_text("\n".join(lines))
        print(f"\n[thrace-report] wrote Markdown: {mdp}")

if __name__ == "__main__":
    main()
