#!/usr/bin/env python3
import json, glob, os, re, math, argparse
from pathlib import Path
import pandas as pd

TAGS = [
    ("shared_plus_tight", r"(shared[_-]?tight)"),
    ("shared_scale",      r"(sharedscale|shared[_-]?scale)"),
    ("tight_scales",      r"(tightscales|tight[_-]?scale|sigma0p02|sigmaf_0p02)"),
    ("sigmaf_0p10",       r"sigmaf[_-]?0p10"),
    ("sigmaf_0p20",       r"sigmaf[_-]?0p20"),
    ("gamma_fix",         r"gamma[_-]?fix"),
    ("gamma_free",        r"gamma[_-]?free"),
]
def infer_tag(fname: str) -> str:
    s=fname.lower()
    for tag,pat in TAGS:
        if re.search(pat, s): return tag
    return "baseline/other"

def load_rows(files):
    rows=[]
    for p in files:
        try:
            J=json.load(open(p))
        except Exception:
            continue
        chi=J.get("chi2", None)
        if chi is None or (isinstance(chi,float) and math.isnan(chi)): continue
        bp=J.get("best_params", {})
        ndof=J.get("ndof", None)
        rows.append({
            "file": os.path.basename(p),
            "chi2": float(chi),
            "chi2_red": (float(chi)/ndof) if isinstance(ndof,(int,float)) and ndof>0 else None,
            "A": bp.get("A"), "f": bp.get("f"), "gamma": bp.get("gamma"), "phi": bp.get("phi"),
            "H0": bp.get("H0"), "Om": bp.get("Om"),
            "tag": infer_tag(os.path.basename(p))
        })
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/*.json", help="Glob of candidate JSONs (quote it)")
    ap.add_argument("--include", default=r"(joint_gamma_|joint_.*shared|joint_.*tight)", help="Regex to keep")
    ap.add_argument("--exclude", default=r"(alpha|resn_alpha|fs8_.*_meta|FROM_SUMMARIES|_meta\.json$)", help="Regex to drop")
    ap.add_argument("--max-chi2", type=float, default=1e6, help="Drop rows with chi2 above this")
    ap.add_argument("--out-csv", default="outputs/chi_summary.csv")
    ap.add_argument("--out-md",  default="outputs/chi_summary.md")
    args=ap.parse_args()

    files=sorted(glob.glob(args.glob))
    inc=re.compile(args.include) if args.include else None
    exc=re.compile(args.exclude) if args.exclude else None
    cand=[p for p in files if (not inc or inc.search(os.path.basename(p))) and (not exc or not exc.search(os.path.basename(p)))]

    rows=load_rows(cand)
    if not rows: raise SystemExit("No usable JSONs after filters. Tweak --glob/--include/--exclude.")
    df=pd.DataFrame(rows)

    # filter absurd χ²
    df=df[(df["chi2"]<=args.max-chi2 if hasattr(args,"max-chi2") else df["chi2"]<=args.max_chi2)].copy()  # handle shell typo safety
    df=df[df["chi2"]<=args.max_chi2].copy()
    if df.empty: raise SystemExit("All rows filtered out by --max-chi2.")

    chi_min=df["chi2"].min()
    df["delta_chi2"]=df["chi2"]-chi_min
    df=df.sort_values(["chi2","file"]).reset_index(drop=True)

    # Per-tag best table
    per=df.loc[df.groupby("tag")["chi2"].idxmin()].sort_values("chi2").reset_index(drop=True)

    # Save CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Markdown (top 12 + per-tag best)
    def fmt(x):
        if x is None or (isinstance(x,float) and (math.isnan(x))): return "—"
        return f"{x:.3f}" if isinstance(x,(int,float)) else str(x)
    lines=[]
    lines.append(f"**Best χ² (filtered):** {chi_min:.3f}\n")
    lines.append("| rank | file | tag | χ² | Δχ² | χ²_red | γ | f | A | φ | H0 | Ωm |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i,r in df.head(12).iterrows():
        lines.append(f"| {i+1} | {r['file']} | {r['tag']} | {r['chi2']:.3f} | {r['delta_chi2']:+.3f} | {fmt(r['chi2_red'])} | {fmt(r['gamma'])} | {fmt(r['f'])} | {fmt(r['A'])} | {fmt(r['phi'])} | {fmt(r['H0'])} | {fmt(r['Om'])} |")
    lines.append("\n**Per-tag best:**\n")
    lines.append("| tag | best_file | χ² | Δχ² |")
    lines.append("|---|---|---:|---:|")
    for _,r in per.iterrows():
        lines.append(f"| {r['tag']} | {r['file']} | {r['chi2']:.3f} | {r['chi2']-chi_min:+.3f} |")
    Path(args.out_md).write_text("\n".join(lines)+"\n")

    # Console print
    print(df.to_string(index=False, max_cols=20, max_rows=30, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {args.out_csv}\nWrote {args.out_md}")

if __name__=="__main__":
    main()
