#!/usr/bin/env python3
import argparse, os, json, pandas as pd, numpy as np

MD_TMPL = """# Phase 15 — Ridge vs Centerline

**Inputs**
- Shell: `{shell_csv}`
- Centerline: `{center_csv}`

**Ridge fit (quadratic about w0):**
- a = {a:.6f}, b = {b:.6f}, **c = {c:.6f}**, R² = {R2:.4f}

**Centerline curvature (Phase-14):**
- **c_center = {c_center:.6f}**

**Offset ΔKϕ = Kϕ_ridge − Kϕ_centerline** (N={n:d})
- mean = {mu:.6f}, std = {sd:.6f}
- min = {mn:.6f}, 25% = {q1:.6f}, median = {med:.6f}, 75% = {q3:.6f}, max = {mx:.6f}

**Figures**
- Ridge field: `{ridge_png}`
- Ridge vs centerline: `{cmp_png}`
"""

TEX_TMPL = r"""\begin{tabular}{l r}
\toprule
Ridge curvature $c$ & %(c).6f \\
Centerline curvature $c_{\mathrm{center}}$ & %(c_center).6f \\
$R^2$ (ridge fit) & %(R2).4f \\
$\mathrm{mean}\,\Delta K_\phi$ & %(mu).6f \\
$\mathrm{std}\,\Delta K_\phi$ & %(sd).6f \\
$\mathrm{median}\,\Delta K_\phi$ & %(med).6f \\
$\mathrm{min}/\mathrm{max}\,\Delta K_\phi$ & %(mn).6f / %(mx).6f \\
\bottomrule
\end{tabular}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge-csv", required=True)
    ap.add_argument("--ridge-fit-json", required=False, default="")
    ap.add_argument("--centerline-csv", required=True)
    ap.add_argument("--ridge-png", required=True)
    ap.add_argument("--cmp-png", required=True)
    ap.add_argument("--shell-csv", required=True)
    ap.add_argument("--outdir", default="outputs/phase15")
    ap.add_argument("--prefix", default="p15")
    ap.add_argument("--c-center", type=float, required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--b", type=float, required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--R2", type=float, required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ridge = pd.read_csv(args.ridge_csv).sort_values("omega2")
    center= pd.read_csv(args.centerline_csv).sort_values("omega2")
    # interpolate centerline to ridge omega2
    Kc = np.interp(ridge["omega2"].values, center["omega2"].values, center["Kphi_mid"].values)
    delta = ridge["Kphi_ridge"].values - Kc
    s = pd.Series(delta).describe()

    md = MD_TMPL.format(
        shell_csv=args.shell_csv, center_csv=args.centerline_csv,
        a=args.a, b=args.b, c=args.c, R2=args.R2,
        c_center=args.c_center,
        n=int(s["count"]), mu=float(s["mean"]), sd=float(s["std"]),
        mn=float(s["min"]), q1=float(np.percentile(delta,25)),
        med=float(np.median(delta)), q3=float(np.percentile(delta,75)),
        mx=float(s["max"]),
        ridge_png=args.ridge_png, cmp_png=args.cmp_png
    )

    tex = TEX_TMPL % dict(
        c=args.c, c_center=args.c_center, R2=args.R2,
        mu=float(s["mean"]), sd=float(s["std"]), med=float(np.median(delta)),
        mn=float(s["min"]), mx=float(s["max"])
    )

    md_path  = os.path.join(args.outdir, f"{args.prefix}_report_card.md")
    tex_path = os.path.join(args.outdir, f"{args.prefix}_report_table.tex")
    with open(md_path, "w") as f: f.write(md)
    with open(tex_path,"w") as f: f.write(tex)

    print("wrote:", md_path)
    print("wrote:", tex_path)

if __name__ == "__main__":
    main()
