#!/usr/bin/env python3
import argparse, json, pathlib, subprocess, pandas as pd

def make_report(policy_map_csv, manifest_json, out_md, out_pdf, resource_path):
    df=pd.read_csv(policy_map_csv)
    man=json.loads(pathlib.Path(manifest_json).read_text())
    rec=man["recommendation"]

    md=[]
    md.append("# Phase-27 — Policy sweep & selection\n")
    md.append(f"- Policy map: `{policy_map_csv}`  \n")
    md.append("Closed-loop worlds are simulated for each policy (R, deadband_frac) under chaotic conditions; ")
    md.append("we report % no-slip across worlds and pick a Pareto-ish recommendation: ")
    md.append("max % no-slip, then min R, then max deadband (calmer action).\n")

    # recommendation
    md.append("\n## Recommendation\n")
    md.append(f"- target: **{rec['target']:.2f}**  \n")
    md.append(f"- R: **{rec['R']}**  \n")
    md.append(f"- deadband_frac (f_h): **{rec['deadband_frac']}**  \n")
    md.append(f"- % no-slip: **{rec['pct_no_slip']:.1f}%**  \n")
    md.append(f"- window: $\\omega^2\\in[{rec['w2_start']:.2f},{rec['w2_end']:.2f}]$, v={rec['v']}  \n")

    # embed plots produced by p27_plot.py
    md.append("\n## Heatmaps\n\n")
    for tgt in sorted(df["target"].unique()):
        md.append(f"![](p27_t{tgt:.2f}_policy_heatmap.png)\n\n")

    md.append("\n## Recommendation (table)\n\n![](p27_recommendation.png)\n\n")

    pathlib.Path(out_md).write_text("".join(md), encoding="utf-8")
    subprocess.run([
        "pandoc","-f","markdown+tex_math_dollars",out_md,"-o",out_pdf,
        "--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path",resource_path
    ], check=True)

def main():
    ap=argparse.ArgumentParser(description="Phase-27 report")
    ap.add_argument("--policy-map-csv", default="outputs/phase27/p27_policy_map.csv")
    ap.add_argument("--manifest-json", default="outputs/phase27/p27_manifest.json")
    ap.add_argument("--outdir", default="outputs/phase27")
    ap.add_argument("--prefix", default="p27")
    args=ap.parse_args()
    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_md=outdir/(args.prefix+"_report.md")
    out_pdf=outdir/(args.prefix+"_report.pdf")
    make_report(args.policy_map_csv, args.manifest_json, str(out_md), str(out_pdf), str(outdir))
    print(json.dumps({"status":"ok","report_md":str(out_md),"report_pdf":str(out_pdf)}, indent=2))

if __name__=="__main__":
    main()
