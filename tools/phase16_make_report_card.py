#!/usr/bin/env python3
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_rows(path):
    rows = json.load(open(path))
    return [r for r in rows if r.get("status") == "ok"]

def as_df(rows):
    cols = ["tag","A_mask_norm","A_trapz_norm","mean_width","A_mask","A_trapz",
            "A_domain","c_center","ridge_c","ridge_R2","d_mean","d_std","d_med",
            "d_min","d_q25","d_q75","d_max","width_png","mask_png","centerline_png",
            "ridge_png","cmp_png"]
    return pd.DataFrame([{k:r.get(k) for k in cols} for r in rows])

def bar_ticks(ax, tags):
    x = np.arange(len(tags))
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right")
    return x

def make_plots(df, outdir, prefix):
    df = df.sort_values("tag")
    tags = df["tag"].astype(str).tolist()

    # 1) A_mask_norm
    fig1, ax1 = plt.subplots(figsize=(8,4.5), dpi=160)
    x = bar_ticks(ax1, tags)
    ax1.bar(x, df["A_mask_norm"].astype(float).values)
    ax1.set_title("Normalized band area (A_mask_norm)")
    ax1.set_ylabel("fraction of domain")
    fig1.tight_layout(); p1 = os.path.join(outdir, f"{prefix}_A_mask_norm.png")
    fig1.savefig(p1); plt.close(fig1)

    # 2) Curvatures
    fig2, ax2 = plt.subplots(figsize=(8,4.5), dpi=160)
    x = np.arange(len(tags)); width = 0.35
    ax2.bar(x - width/2, df["c_center"].astype(float).values, width, label="centerline c")
    ax2.bar(x + width/2, df["ridge_c"].astype(float).values, width, label="ridge c")
    ax2.set_title("Curvature comparison (centerline vs ridge)")
    ax2.set_xticks(x); ax2.set_xticklabels(tags, rotation=20, ha="right")
    ax2.set_ylabel("curvature"); ax2.legend(frameon=False)
    fig2.tight_layout(); p2 = os.path.join(outdir, f"{prefix}_curvatures.png")
    fig2.savefig(p2); plt.close(fig2)

    # 3) Mean ΔKphi
    fig3, ax3 = plt.subplots(figsize=(8,4.5), dpi=160)
    x = bar_ticks(ax3, tags)
    ax3.bar(x, df["d_mean"].astype(float).values)
    ax3.set_title("Ridge–centerline offset (mean ΔKϕ)")
    ax3.set_ylabel("ΔKϕ")
    fig3.tight_layout(); p3 = os.path.join(outdir, f"{prefix}_delta_mean.png")
    fig3.savefig(p3); plt.close(fig3)

    return p1, p2, p3

def make_tables(df, outdir, prefix):
    md = ["# Phase 16 — Multi-shell Summary\n",
          "| tag | A_mask_norm | mean_width | A_mask | A_trapz | c_center | ridge_c | R2_ridge | mean ΔKϕ | std ΔKϕ |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _,r in df.iterrows():
        md.append(f"| {r.tag} | {r.A_mask_norm:.4f} | {r.mean_width:.4f} | {r.A_mask:.6f} | {r.A_trapz:.6f} | "
                  f"{r.c_center:.4f} | {r.ridge_c:.4f} | {r.ridge_R2:.3f} | {r.d_mean:.4f} | {r.d_std:.4f} |")
    md_path = os.path.join(outdir, f"{prefix}_report_card.md")
    with open(md_path, "w") as f: f.write("\n".join(md) + "\n")

    tex = [r"\begin{tabular}{l r r r r r r r r r}", r"\toprule",
           r"tag & $A_{\mathrm{mask,norm}}$ & $\overline{w}$ & $A_{\mathrm{mask}}$ & $A_{\mathrm{trapz}}$ & $c_{\mathrm{center}}$ & $c_{\mathrm{ridge}}$ & $R^2_{\mathrm{ridge}}$ & $\mathrm{mean}\,\Delta K_\phi$ & $\mathrm{std}\,\Delta K_\phi$ \\",
           r"\midrule"]
    for _,r in df.iterrows():
        tex.append(f"{r.tag} & {r.A_mask_norm:.4f} & {r.mean_width:.4f} & {r.A_mask:.6f} & {r.A_trapz:.6f} & "
                   f"{r.c_center:.4f} & {r.ridge_c:.4f} & {r.ridge_R2:.3f} & {r.d_mean:.4f} & {r.d_std:.4f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(outdir, f"{prefix}_table.tex")
    with open(tex_path, "w") as f: f.write("\n".join(tex) + "\n")
    return md_path, tex_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--outdir", default="outputs/phase16")
    ap.add_argument("--prefix", default="p16")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.summary_json)
    if not rows:
        print("No OK rows in summary; nothing to report."); return 0
    df = as_df(rows)

    p1, p2, p3 = make_plots(df, args.outdir, args.prefix)
    md_path, tex_path = make_tables(df, args.outdir, args.prefix)

    print("wrote:", p1); print("wrote:", p2); print("wrote:", p3)
    print("wrote:", md_path); print("wrote:", tex_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
