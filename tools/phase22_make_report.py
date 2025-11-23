#!/usr/bin/env python3
import argparse, pathlib, csv, shutil, subprocess, json, textwrap

def has_pandoc():
    return shutil.which("pandoc") is not None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertainty-csv", required=True)
    ap.add_argument("--bands-json", required=True)
    ap.add_argument("--bands-png", required=True)
    ap.add_argument("--sigma-png", required=True)
    ap.add_argument("--outdir", default="outputs/phase22")
    ap.add_argument("--prefix", default="p22")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    md = outdir / f"{args.prefix}_report.md"
    pdf = outdir / f"{args.prefix}_report.pdf"

    # A tiny table (head only) + sample rows
    head = []
    rows = []
    with open(args.uncertainty_csv, newline="") as f:
        r = csv.DictReader(f)
        head = r.fieldnames
        for i, row in enumerate(r):
            if i < 10: rows.append(row)

    with open(md, "w") as f:
        f.write("# Phase-22 — Uncertainty Bands\n\n")
        f.write("**Protocol:** Lock → Verify → Style. This document is the Verify pass.\n\n")
        f.write("## Inputs\n")
        f.write(f"- Guardbands: from Phase-21\n")
        f.write(f"- Bands JSON: `{args.bands_json}`\n\n")
        f.write("## Method (summary)\n")
        f.write(textwrap.dedent("""
        - For each guardband point, estimate the local slope \(\\partial s_i / \\partial K_\\phi\) via a central finite difference.
        - Bootstrap the simulator at the centerline \(K_\\phi^{mid}\) to get \(\\sigma_{s_i}\).
        - Propagate to \(\\sigma_{K_\\phi} \\approx \\sigma_{s_i} / |\\partial s_i / \\partial K_\\phi|\).
        - Form 95% CIs for \(K_\\phi\) and band width; aggregate per target to plot ribbons.
        """))
        f.write("\n## Plots\n")
        f.write(f"![Bands with 95% CI]({args.bands_png})\n\n")
        f.write(f"![Local σ in $K_\\phi$]({args.sigma_png})\n\n")
        f.write("## Sample of `p22_uncertainty.csv`\n\n")
        # Markdown table (head + first 10)
        f.write("| " + " | ".join(head) + " |\n")
        f.write("|" + "|".join(["---"]*len(head)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in head) + " |\n")

    if has_pandoc():
        subprocess.run(["pandoc", str(md), "-o", str(pdf)], check=False)

    print(json.dumps({
        "report_md": str(md),
        "report_pdf": str(pdf) if pdf.exists() else None
    }, indent=2))

if __name__ == "__main__":
    main()
