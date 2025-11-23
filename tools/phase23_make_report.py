#!/usr/bin/env python3
import csv, pathlib, subprocess, re

def sanitize(md: str) -> str:
    md = "".join(ch for ch in md if ch in ("\t","\n","\r") or 32 <= ord(ch) <= 126)
    md = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", md, flags=re.DOTALL)
    md = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"($\1$)", md, flags=re.DOTALL)
    for pat in (r"\\omega\^2", r"K\\_\\phi", r"s\\_i", r"\\sigma"):
        md = re.sub(rf"(?<!\$)({pat})(?!\$)", r"$\1$", md)
    return md

def make_report(unc_csv, drift_csv, out_md, out_pdf, resource_path):
    with open(unc_csv, newline="") as f:
        urows = list(csv.DictReader(f))
    with open(drift_csv, newline="") as f:
        drows = list(csv.DictReader(f))

    targets = sorted({float(r["target"]) for r in urows})
    w2s     = sorted({float(r["omega2"]) for r in urows})

    md=[]
    md.append("# Phase-23 — Drift Surface\n")
    md.append(f"- Source uncertainty: `{unc_csv}`  \n")
    md.append(f"- Drift CSV: `{drift_csv}`  \n")
    md.append(f"- Targets: `{[round(t,3) for t in targets]}`  \n")
    md.append(f"- ω² grid: `{[round(w,3) for w in w2s]}`  (|grid|={len(w2s)})\n")

    md.append("\n## Method\n")
    md.append("For each target, we compute midline $K_\\phi$ and estimate the drift ")
    md.append("$dK_\\phi/d\\omega^2$ by finite differences across $\\omega^2$. ")
    md.append("Central differencing is used when possible; one-sided at edges.\n")

    show = drows[:min(8,len(drows))]
    if show:
        md.append("\n## Example drift rows\n")
        md.append("| target | ω² | Kφ(mid) | dKφ/dω² |\n|---:|---:|---:|---:|\n")
        for r in show:
            md.append(f"| {float(r['target']):.3f} | {float(r['omega2']):.3f} | {float(r['Kphi_mid']):.4f} | {float(r['dK_domega2']):.4f} |\n")

    for name, title in [
        ("p23_kmid_vs_w2.png", "Midline Kφ vs ω²"),
        ("p23_dkdw2_vs_w2.png", "Drift dKφ/dω² per target"),
        ("p23_drift_heatmap.png", "Drift magnitude heatmap"),
    ]:
        p = pathlib.Path(resource_path)/name
        if p.exists():
            md.append(f"\n## {title}\n\n![]({name})\n")

    md_text=sanitize("".join(md))
    pathlib.Path(out_md).write_text(md_text, encoding="utf-8")

    cmd=[
        "pandoc","-f","markdown+tex_math_dollars",out_md,
        "-o", out_pdf,"--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path", str(resource_path)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Wrote:", out_pdf)

def main():
    import argparse
    ap=argparse.ArgumentParser(description="Phase-23 report")
    ap.add_argument("--uncertainty-csv", default="outputs/phase22/p22_uncertainty.csv")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--out-md", default="outputs/phase23/p23_report.md")
    ap.add_argument("--out-pdf", default="outputs/phase23/p23_report.pdf")
    ap.add_argument("--resource-path", default="outputs/phase23")
    args=ap.parse_args()
    make_report(args.uncertainty_csv, args.drift_csv, args.out_md, args.out_pdf, args.resource_path)

if __name__=="__main__":
    main()
