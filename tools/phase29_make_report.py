#!/usr/bin/env python3
import argparse, json, pathlib, subprocess, pandas as pd

def make_report(endurance_csv, aggregate_csv, manifest_json, out_md, out_pdf, resource_path):
    df=pd.read_csv(endurance_csv)
    agg = pd.read_csv(aggregate_csv) if pathlib.Path(aggregate_csv).exists() else None
    man=json.loads(pathlib.Path(manifest_json).read_text()) if pathlib.Path(manifest_json).exists() else {"params":{}}

    total=len(df); ok=int((df["no_slip"]=="yes").sum()) if total else 0
    pct = 100.0*ok/total if total else 0.0
    pol=man.get("policy",{}); pars=man.get("params",{})
    md=[]
    md.append("# Phase-29 — Bias-rejecting controller (soft integrator)\n")
    md.append(f"- Endurance runs: `{endurance_csv}`  \n")
    if agg is not None and not agg.empty:
        md.append(f"- Aggregates: `{aggregate_csv}`  \n")
    md.append(f"- Policy (base): R={pol.get('R','?')}, f_h={pol.get('deadband_frac','?')}, slip_frac={pol.get('slip_frac','?')}  \n")
    md.append(f"- Integrator: beta={pars.get('beta','?')}, eta={pars.get('eta','?')}, leak={pars.get('leak','?')}  \n")
    md.append(f"- Window: omega^2 in [{pol.get('w2_start','?')}, {pol.get('w2_end','?')}], v={pars.get('v','?')}  \n")

    md.append("\n## Results\n")
    md.append(f"- Overall no-slip: **{pct:.1f}%** ({ok}/{total})\n")
    if agg is not None and not agg.empty:
        md.append("\n### Per-target aggregates (medians)\n\n")
        md.append("| target | % no-slip | effort_L1_med | chatter_med | mean_abs_err_med | t_first_slip_med | recovery_median_s_med |\n")
        md.append("|---:|---:|---:|---:|---:|---:|---:|\n")
        for _,r in agg.iterrows():
            md.append(f"| {float(r['target']):.2f} | {float(r['%no_slip']):.1f} | "
                      f"{r['effort_L1_med']} | {r['chatter_med']} | {r['mean_abs_err_med']} | "
                      f"{r['t_first_slip_med']} | {r['recovery_median_s_med']} |\n")

    md.append("\n## Plots\n\n")
    for p in ["p29_noslip_by_target.png","p29_t_first_slip_hist.png","p29_effort_L1_hist.png","p29_chatter_hist.png","p29_mean_abs_err_hist.png"]:
        if pathlib.Path(resource_path, p).exists():
            md.append(f"![]({p})\n\n")

    pathlib.Path(out_md).write_text("".join(md), encoding="utf-8")
    subprocess.run([
        "pandoc","-f","markdown+tex_math_dollars",out_md,"-o",out_pdf,
        "--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path",resource_path
    ], check=True)

def main():
    import argparse, pathlib, json
    ap=argparse.ArgumentParser(description="Phase-29 report")
    ap.add_argument("--endurance-csv", default="outputs/phase29/p29_endurance.csv")
    ap.add_argument("--aggregate-csv", default="outputs/phase29/p29_aggregate.csv")
    ap.add_argument("--manifest-json", default="outputs/phase29/p29_manifest.json")
    ap.add_argument("--outdir", default="outputs/phase29")
    ap.add_argument("--prefix", default="p29")
    args=ap.parse_args()
    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_md=outdir/(args.prefix+"_report.md"); out_pdf=outdir/(args.prefix+"_report.pdf")
    make_report(args.endurance_csv, args.aggregate_csv, args.manifest_json, str(out_md), str(out_pdf), str(outdir))
    print(json.dumps({"status":"ok","report_md":str(out_md),"report_pdf":str(out_pdf)}, indent=2))

if __name__=="__main__":
    main()
