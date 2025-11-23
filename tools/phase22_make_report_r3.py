#!/usr/bin/env python3
import csv, json, pathlib, subprocess, sys, re
from collections import defaultdict

def sanitize_math(md: str) -> str:
    md = "".join(ch for ch in md if ch in ("\t","\n","\r") or 32 <= ord(ch) <= 126)
    # normalize TeX bracket math to dollar math
    md = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", md, flags=re.DOTALL)
    md = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"($\1$)", md, flags=re.DOTALL)
    # wrap common tokens
    for pat in (r"\\omega\^2", r"K\\_\\phi", r"s\\_i", r"\\sigma"):
        md = re.sub(rf"(?<!\$)({pat})(?!\$)", r"$\1$", md)
    # fix any leftover weird mid token
    md = md.replace(r"(\[^{mid}$", r"($^{mid}$)")
    md = md.replace(r"(\[^{mid}\)", r"($^{mid}$)")
    return md

def load_bands_list(path: str):
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for k in ("bands","data","rows"):
            if k in obj and isinstance(obj[k], list): return obj[k]
        return [obj]
    return []

def fmt_stats(xs):
    if not xs: return "n/a"
    xs = sorted(xs)
    n=len(xs)
    mean = sum(xs)/n
    med  = xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])
    p05  = xs[max(0, int(0.05*(n-1)))]
    p95  = xs[int(0.95*(n-1))]
    return f"n={n}, mean={mean:.4g}, median={med:.4g}, p05={p05:.4g}, p95={p95:.4g}"

def make_coverage_table(rows):
    targets = sorted({float(r["target"]) for r in rows})
    w2s     = sorted({float(r["omega2"]) for r in rows})
    have = {(float(r["target"]), float(r["omega2"])) for r in rows}
    # header
    out = []
    head = "| target \\ ω² | " + " | ".join(f"{w:.2f}" for w in w2s) + " |\n"
    sep  = "|:-----------:|" + "|".join([":--:"]*len(w2s)) + "|\n"
    out.append(head); out.append(sep)
    for t in targets:
        row = [f"{t:.2f}"] + [("✓" if (t,w) in have else "✗") for w in w2s]
        out.append("| " + " | ".join(row) + " |\n")
    return "".join(out)

def make_report(unc_csv, bands_json, out_md, out_pdf, resource_path):
    with open(unc_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    bands_rows = load_bands_list(bands_json)

    # derive stats
    targets = sorted({float(r["target"]) for r in rows})
    w2s     = sorted({float(r["omega2"]) for r in rows})
    bw      = [float(r["band_width"]) for r in rows if r.get("band_width")]
    bw_lo   = [float(r["band_width_CI_lo"]) for r in rows if r.get("band_width_CI_lo")]
    bw_hi   = [float(r["band_width_CI_hi"]) for r in rows if r.get("band_width_CI_hi")]

    # select neat example subset (first 10 by ω² then target)
    show = sorted(rows, key=lambda r: (float(r["omega2"]), float(r["target"])))[:10]

    md = []
    md.append("# Phase-22 — Uncertainty Bands (r3)\n")
    md.append(f"- Uncertainty CSV: `{unc_csv}`  \n")
    md.append(f"- Bands JSON: `{bands_json}`  \n")
    md.append(f"- Rows: **{len(rows)}**  \n")
    md.append(f"- Targets: `{[round(t,3) for t in targets]}`  \n")
    md.append(f"- ω² grid: `{[round(w,3) for w in w2s]}`  (|grid|={len(w2s)})\n")

    md.append("\n## Band width summary\n")
    md.append(f"- band_width: {fmt_stats(bw)}\n")
    if bw_lo: md.append(f"- band_width_CI_lo: {fmt_stats(bw_lo)}\n")
    if bw_hi: md.append(f"- band_width_CI_hi: {fmt_stats(bw_hi)}\n")

    md.append("\n## Example rows\n")
    if show:
        md.append("| target | ω² | Kφ_lo | Kφ_hi | width | σ(Kφ) | s_i(mid) |\n")
        md.append("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in show:
            md.append(
                f"| {float(r['target']):.3f} | {float(r['omega2']):.3f} "
                f"| {float(r['Kphi_lo']):.3f} | {float(r['Kphi_hi']):.3f} "
                f"| {float(r.get('band_width','nan')):.3f} "
                f"| {float(r.get('sigma_Kphi','nan')):.3f} "
                f"| {float(r.get('si_mid','nan')):.3f} |\n"
            )
    else:
        md.append("_(no rows)_\n")

    md.append("\n## Coverage grid (✓ band exists, ✗ none)\n")
    md.append(make_coverage_table(rows))

    # method note
    md.append("\n## Method\n")
    md.append(
        "We estimate the local slope at the midpoint as "
        "($\\partial s_i/\\partial K_\\phi$), bootstrap $s_i$, and propagate to "
        "$\\sigma_{K_\\phi}$ via linear error propagation. Flat slopes inflate "
        "$\\sigma_{K_\\phi}$ by design.\n"
    )

    # plots (shown if present)
    bands_png = pathlib.Path(resource_path)/"p22_bands_CI.png"
    sigma_png = pathlib.Path(resource_path)/"p22_sigma_map.png"
    if bands_png.exists(): md.append(f"\n## Bands (with CIs)\n\n![](p22_bands_CI.png)\n")
    if sigma_png.exists(): md.append(f"\n## σ(Kφ) map\n\n![](p22_sigma_map.png)\n")

    md_text = sanitize_math("\n".join(md))
    pathlib.Path(out_md).write_text(md_text, encoding="utf-8")

    # pandoc with resource-path so images embed
    cmd = [
        "pandoc",
        "-f", "markdown+tex_math_dollars",
        out_md,
        "-o", out_pdf,
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=20mm",
        "--resource-path", str(resource_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Wrote:", out_pdf)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Phase-22 report (r3, tidy)")
    ap.add_argument("--uncertainty-csv", required=True)
    ap.add_argument("--bands-json", required=True)
    ap.add_argument("--out-md", default="outputs/phase22/p22r3_report.md")
    ap.add_argument("--out-pdf", default="outputs/phase22/p22r3_report.pdf")
    ap.add_argument("--resource-path", default="outputs/phase22")
    args = ap.parse_args()
    make_report(args.uncertainty_csv, args.bands_json, args.out_md, args.out_pdf, args.resource_path)

if __name__ == "__main__":
    main()
