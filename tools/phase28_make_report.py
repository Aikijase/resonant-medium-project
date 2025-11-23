#!/usr/bin/env python3
import argparse, json, pathlib, subprocess
import pandas as pd

def make_report(endurance_csv, aggregate_csv, manifest_json, out_md, out_pdf, resource_path):
    # Load inputs (endurance is required)
    df = pd.read_csv(endurance_csv)

    agg = None
    agg_path = pathlib.Path(aggregate_csv)
    if agg_path.exists():
        try:
            tmp = pd.read_csv(aggregate_csv)
            if not tmp.empty:
                agg = tmp
        except Exception:
            agg = None

    man = json.loads(pathlib.Path(manifest_json).read_text()) if pathlib.Path(manifest_json).exists() else {"params":{}}
    total = len(df)
    ok = int((df["no_slip"] == "yes").sum()) if total else 0
    pct = 100.0 * ok / total if total else 0.0

    pol = man.get("policy", {})
    v_used = man.get("params", {}).get("v", "?")

    md = []
    md.append("# Phase-28 — Endurance and Disturbance-Rejection\n")
    md.append(f"- Endurance runs: `{endurance_csv}`  \n")
    if agg is not None and not agg.empty:
        md.append(f"- Aggregates: `{aggregate_csv}`  \n")

    md.append(f"- Policy used: R={pol.get('R','?')}, deadband_frac={pol.get('deadband_frac','?')}, slip_frac={pol.get('slip_frac','?')}  \n")
    md.append(f"- Window: omega^2 in [{pol.get('w2_start','?')}, {pol.get('w2_end','?')}] at v={v_used}  \n")

    md.append("\n## Results\n")
    md.append(f"- Overall no-slip rate: **{pct:.1f}%** ({ok}/{total})\n")

    if agg is not None and not agg.empty:
        md.append("\n### Per-target aggregates (medians)\n\n")
        md.append("| target | % no-slip | effort_L1_med | chatter_med | mean_abs_err_med | t_first_slip_med | recovery_median_s_med |\n")
        md.append("|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in agg.iterrows():
            md.append(
                f"| {float(r['target']):.2f} | {float(r['%no_slip']):.1f} | "
                f"{r['effort_L1_med']} | {r['chatter_med']} | {r['mean_abs_err_med']} | "
                f"{r['t_first_slip_med']} | {r['recovery_median_s_med']} |\n"
            )

    # Plots (embed only if files exist)
    md.append("\n## Plots\n\n")
    for p in ["p28_noslip_by_target.png", "p28_t_first_slip_hist.png", "p28_effort_L1_hist.png", "p28_chatter_hist.png"]:
        if pathlib.Path(resource_path, p).exists():
            md.append(f"![]({p})\n\n")

    pathlib.Path(out_md).write_text("".join(md), encoding="utf-8")

    subprocess.run([
        "pandoc","-f","markdown+tex_math_dollars",out_md,"-o",out_pdf,
        "--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path",resource_path
    ], check=True)

def main():
    ap = argparse.ArgumentParser(description="Phase-28 report")
    ap.add_argument("--endurance-csv", default="outputs/phase28/p28_endurance.csv")
    ap.add_argument("--aggregate-csv", default="outputs/phase28/p28_aggregate.csv")
    ap.add_argument("--manifest-json", default="outputs/phase28/p28_manifest.json")
    ap.add_argument("--outdir", default="outputs/phase28")
    ap.add_argument("--prefix", default="p28")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_md = outdir / (args.prefix + "_report.md")
    out_pdf = outdir / (args.prefix + "_report.pdf")

    make_report(args.endurance_csv, args.aggregate_csv, args.manifest_json, str(out_md), str(out_pdf), str(outdir))
    print(json.dumps({"status":"ok","report_md":str(out_md),"report_pdf":str(out_pdf)}, indent=2))

if __name__ == "__main__":
    main()
