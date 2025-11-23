#!/usr/bin/env python3
import argparse, json, pathlib, csv, subprocess, pandas as pd

def make_report(worlds_csv, out_md, out_pdf, resource_path):
    df=pd.read_csv(worlds_csv)
    total=len(df)
    no_slip=(df["slipped"]=="no").sum()
    pct = 100.0*no_slip/total if total>0 else 0.0
    by=df.groupby("target")["slipped"].apply(lambda s:(s=="no").mean()*100).reset_index(name="pct_no_slip")

    md=[]
    md.append("# Phase-26 — Full loop stress test\n")
    md.append(f"- Worlds: `{worlds_csv}`  \n")
    md.append("## Setup\n")
    md.append("Closed-loop controller (rate-limit + deadband) tracks $K_\\phi$ to $K_{\\text{mid}}(\\omega^2)$. ")
    md.append("Chaotic perturbations: colored noise on $\\omega^2$ and measurement, ")
    md.append("occasional band shrink, and random velocity bursts. Slip when ")
    md.append("$|K_{\\text{mid}}-K_\\phi|\\ge f_s\\cdot \\tfrac12\\,\\mathrm{band\\_width}$ (measured).\n")

    md.append("## Results\n")
    md.append(f"- Overall no-slip rate: **{pct:.1f}%** ({no_slip}/{total})\n\n")
    md.append("| target | % no-slip |\n|---:|---:|\n")
    for _,r in by.iterrows():
        md.append(f"| {float(r['target']):.2f} | {float(r['pct_no_slip']):.1f} |\n")

    # embed known plots if present
    plots = [
        "p26_noslip_bar.png",
        "p26_t_slip_hist.png"
    ]
    md.append("\n## Plots\n\n")
    for p in plots:
        md.append(f"![]({p})\n\n")

    pathlib.Path(out_md).write_text("".join(md), encoding="utf-8")
    subprocess.run([
        "pandoc","-f","markdown+tex_math_dollars",out_md,"-o",out_pdf,
        "--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path",resource_path
    ], check=True)

def main():
    ap=argparse.ArgumentParser(description="Phase-26 report")
    ap.add_argument("--worlds-csv", default="outputs/phase26/p26_worlds.csv")
    ap.add_argument("--outdir", default="outputs/phase26")
    ap.add_argument("--prefix", default="p26")
    args=ap.parse_args()
    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_md=outdir/(args.prefix+"_report.md")
    out_pdf=outdir/(args.prefix+"_report.pdf")
    make_report(args.worlds_csv, str(out_md), str(out_pdf), str(outdir))
    print(json.dumps({"status":"ok","report_md":str(out_md),"report_pdf":str(out_pdf)}, indent=2))

if __name__=="__main__":
    main()
