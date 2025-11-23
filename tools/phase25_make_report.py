#!/usr/bin/env python3
import argparse, json, pathlib, csv, subprocess

def read_summary(csv_path):
    rows=[]
    with open(csv_path) as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def make_report(sum_csv, out_md, out_pdf, resource_path):
    rows=read_summary(sum_csv)
    md=[]
    md.append("# Phase-25 — Controller policy (rate-limit + hysteresis) & time-to-slip\n")
    md.append(f"- Summary: `{sum_csv}`  \n")
    md.append("## Method\n")
    md.append("We track $K_\\phi(t)$ toward the band midpoint $K_{\\text{mid}}(\\omega^2)$ from Phase-23.\n")
    md.append("The allowable error uses Phase-22 band width: half-width $=\\tfrac12\\,\\mathrm{band\\_width}$.\n")
    md.append("A deadband $h=f_h\\cdot \\tfrac12\\,\\mathrm{band\\_width}$ disables actuation for small errors.\n")
    md.append("The actuator is rate-limited by $R=\\max|dK_\\phi/dt|$. We declare slip when\n")
    md.append("$|e(t)|=|K_{\\text{mid}}-K_\\phi| \\ge f_s\\cdot \\tfrac12\\,\\mathrm{band\\_width}$.\n")

    if rows:
        md.append("\n## Runs\n\n")
        md.append("| target | R | v | w2_start | w2_end | slipped | t_slip | w2_slip |\n")
        md.append("|---:|---:|---:|---:|---:|:---:|---:|---:|\n")
        for r in rows:
            target = float(r['target'])
            R = float(r['R'])
            v = float(r['v'])
            w2_start = float(r['w2_start'])
            w2_end = float(r['w2_end'])
            slipped = r['slipped']
            t_slip = r['t_slip'] if slipped == 'yes' else ''
            w2_slip = r['w2_slip'] if slipped == 'yes' else ''
            md.append(f"| {target:.2f} | {R:g} | {v:g} | {w2_start:.2f} | {w2_end:.2f} | "
                      f"{slipped} | {t_slip} | {w2_slip} |\n")

    # try to embed any *_plot.png
    rp=pathlib.Path(resource_path)
    plots=sorted([p for p in rp.glob("p25_*_plot.png")])
    if plots:
        md.append("\n## Example time series\n\n")
        for p in plots[:6]:
            md.append(f"![]({p.name})\n\n")

    content="".join(md)
    pathlib.Path(out_md).write_text(content, encoding="utf-8")
    cmd=["pandoc","-f","markdown+tex_math_dollars",out_md,"-o",out_pdf,
         "--pdf-engine","xelatex","-V","geometry:margin=20mm",
         "--resource-path",resource_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Wrote:", out_pdf)

def main():
    ap=argparse.ArgumentParser(description="Phase-25 report builder")
    ap.add_argument("--summary-csv", default="outputs/phase25/p25_summary.csv")
    ap.add_argument("--outdir", default="outputs/phase25")
    ap.add_argument("--prefix", default="p25")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_md = outdir/(args.prefix+"_report.md")
    out_pdf = outdir/(args.prefix+"_report.pdf")
    make_report(args.summary_csv, str(out_md), str(out_pdf), str(outdir))

    print(json.dumps({"status":"ok","report_md":str(out_md),"report_pdf":str(out_pdf)}, indent=2))

if __name__=="__main__":
    main()
