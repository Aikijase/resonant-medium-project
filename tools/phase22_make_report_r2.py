#!/usr/bin/env python3
import csv, json, pathlib, subprocess, sys, re
from collections import Counter

def sanitize_math(md_text: str) -> str:
    # keep printable ASCII + tabs/newlines/CR
    md_text = "".join(ch for ch in md_text if ch in ("\t","\n","\r") or 32 <= ord(ch) <= 126)
    # fix the notorious (\[^{mid}$) / (\[^{mid}\)) → ($^{mid}$)
    md_text = md_text.replace(r"(\[^{mid}$", r"($^{mid}$)")
    md_text = md_text.replace(r"(\[^{mid}\)", r"($^{mid}$)")
    # normalize TeX bracket math to dollar math
    md_text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", md_text, flags=re.DOTALL)
    md_text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"($\1$)", md_text, flags=re.DOTALL)
    # wrap common bare tokens
    for pat in (r"\\omega\^2", r"K\\_\\phi", r"s\\_i"):
        md_text = re.sub(rf"(?<!\$)({pat})(?!\$)", r"$\1$", md_text)
    return md_text

def load_bands_obj(path: str):
    with open(path) as f:
        obj = json.load(f)
    # Accept list or dict
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, dict):
        # common keys: maybe stored as {"bands":[...]} or so
        for k in ("bands", "data", "rows"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # fallback: treat dict itself as a single row list
        return [obj]
    else:
        return []

def make_report(unc_csv: str, bands_json: str, out_md: str, out_pdf: str):
    # CSV
    with open(unc_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    # JSON (robust)
    bands_rows = load_bands_obj(bands_json)

    # derive targets and ω² coverage from CSV (robust)
    targets = sorted({float(r["target"]) for r in rows})
    w2s     = sorted({float(r["omega2"]) for r in rows})
    # count rows per target
    per_target = Counter(float(r["target"]) for r in rows)

    # example rows
    show = rows[: min(5, len(rows))]

    # basic stats
    bw = [float(r["band_width"]) for r in rows if r.get("band_width") not in (None,"")]
    bw_ci_lo = [float(r["band_width_CI_lo"]) for r in rows if r.get("band_width_CI_lo") not in (None,"")]
    bw_ci_hi = [float(r["band_width_CI_hi"]) for r in rows if r.get("band_width_CI_hi") not in (None,"")]

    def fmt_stats(xs):
        if not xs: return "n/a"
        xs_sorted = sorted(xs)
        n=len(xs_sorted)
        p50 = xs_sorted[n//2] if n%2 else 0.5*(xs_sorted[n//2-1]+xs_sorted[n//2])
        p05 = xs_sorted[max(0, int(0.05*(n-1)))]
        p95 = xs_sorted[int(0.95*(n-1))]
        return f"n={n}, mean={sum(xs_sorted)/n:.4g}, median={p50:.4g}, p05={p05:.4g}, p95={p95:.4g}"

    # build markdown
    md = []
    md.append("# Phase-22 — Uncertainty Bands (r2)\n")
    md.append(f"- Uncertainty CSV: `{unc_csv}`  \n")
    md.append(f"- Bands JSON: `{bands_json}`  \n")
    md.append(f"- Rows: **{len(rows)}**\n")
    md.append(f"- Targets: `{[round(t,4) for t in targets]}`\n")
    md.append(f"- ω² grid: `{[round(w,4) for w in w2s]}`  (|grid|={len(w2s)})\n")

    # per-target coverage table (markdown)
    md.append("\n## Coverage by target\n")
    md.append("| target | rows |\n|---:|---:|\n")
    for t in targets:
        md.append(f"| {t:.4g} | {per_target.get(t,0)} |\n")

    # summary stats
    md.append("\n## Band width summary\n")
    md.append(f"- band_width: {fmt_stats(bw)}\n")
    if bw_ci_lo and bw_ci_hi:
        md.append(f"- band_width_CI_lo: {fmt_stats(bw_ci_lo)}\n")
        md.append(f"- band_width_CI_hi: {fmt_stats(bw_ci_hi)}\n")

    # example rows
    md.append("\n## Example rows\n")
    if show:
        md.append("| target | ω² | Kφ_lo | Kφ_hi | width | σ(Kφ) |\n|---:|---:|---:|---:|---:|---:|\n")
        for r in show:
            md.append(
                f"| {float(r['target']):.3f} | {float(r['omega2']):.3f} "
                f"| {float(r['Kphi_lo']):.3f} | {float(r['Kphi_hi']):.3f} "
                f"| {float(r.get('band_width', 'nan')):.3f} "
                f"| {float(r.get('sigma_Kphi','nan')):.3f} |\n"
            )
    else:
        md.append("_(no rows)_\n")

    # math note
    md.append("\n## Notes on method\n")
    md.append(
        "We estimate the local slope at the midpoint as "
        "($\\partial s_i/\\partial K_\\phi$), and propagate bootstrap variance "
        "in $s_i$ to uncertainty in $K_\\phi$ via linear error propagation. "
        "Flat slopes inflate $\\sigma_{K_\\phi}$ by design.\n"
    )

    # include plots if present
    bands_png = pathlib.Path(out_md).with_name("p22_bands_CI.png")
    sigma_png = pathlib.Path(out_md).with_name("p22_sigma_map.png")
    if bands_png.exists():
        md.append(f"\n## Bands (with CIs)\n\n![]({bands_png.name})\n")
    if sigma_png.exists():
        md.append(f"\n## σ(Kφ) map\n\n![]({sigma_png.name})\n")

    # sanitize + write
    md_text = sanitize_math("\n".join(md))
    pathlib.Path(out_md).write_text(md_text, encoding="utf-8")

    # render to PDF via pandoc
    cmd = [
        "pandoc",
        "-f", "markdown+tex_math_dollars",
        out_md,
        "-o", out_pdf,
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=20mm",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Wrote:", out_pdf)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Phase-22 report builder (r2 robust)")
    ap.add_argument("--uncertainty-csv", required=True)
    ap.add_argument("--bands-json", required=True)
    ap.add_argument("--out-md", default="outputs/phase22/p22r2_report.md")
    ap.add_argument("--out-pdf", default="outputs/phase22/p22r2_report.pdf")
    args = ap.parse_args()
    make_report(args.uncertainty_csv, args.bands_json, args.out_md, args.out_pdf)

if __name__ == "__main__":
    main()
