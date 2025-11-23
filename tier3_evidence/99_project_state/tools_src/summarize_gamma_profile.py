#!/usr/bin/env python3
import json, glob, os, re, math, statistics as stats
from pathlib import Path

OUTDIR = Path("outputs")
PROFILE_GLOB = str(OUTDIR / "joint_gamma_fix_*.json")
MD_OUT = OUTDIR / "gamma_profile_summary.md"
CSV_OUT = OUTDIR / "gamma_profile_table.csv"
LATEX_OUT = OUTDIR / "gamma_profile_table.tex"

def extract_gamma(J, fname):
    for path in [("best_params","gamma"), ("priors","gamma","mean"), ("fit","gamma_value")]:
        d=J
        try:
            for k in path: d=d[k]
            if isinstance(d,(int,float)): return float(d)
        except Exception: pass
    m=re.search(r"joint_gamma_fix_([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)", os.path.basename(fname))
    return float(m.group(1).replace("p",".")) if m else None

def load_rows():
    rows=[]
    for p in sorted(glob.glob(PROFILE_GLOB)):
        try:
            J=json.load(open(p))
            g=extract_gamma(J, p); chi=J.get("chi2", None)
            if g is None or chi is None: continue
            rows.append((g, float(chi), os.path.basename(p)))
        except Exception: pass
    if not rows: raise SystemExit("No profile jsons.")
    # dedupe by gamma → keep min χ²
    by={}
    for g,chi,b in rows:
        if (g not in by) or chi<by[g][0]: by[g]=(chi,b)
    gam=sorted(by.keys())
    chis=[by[g][0] for g in gam]
    return gam, chis

def ci_from_profile(g, d, level):
    # piecewise-linear interpolation around the minimum
    import numpy as np
    gmin = g[d.index(min(d))]
    # left
    L=float('nan')
    for i in range(len(g)-1,0,-1):
        if g[i-1]<=gmin<=g[i]: break
    for i in range(g.index(gmin),0,-1):
        g0,g1=g[i-1],g[i]; d0,d1=d[i-1],d[i]
        if (d0-level)*(d1-level)<=0 and g0!=g1:
            t=(level-d0)/(d1-d0) if d1!=d0 else 0.0; L=g0+t*(g1-g0); break
    # right
    R=float('nan')
    for i in range(g.index(gmin), len(g)-1):
        g0,g1=g[i],g[i+1]; d0,d1=d[i],d[i+1]
        if (d0-level)*(d1-level)<=0 and g0!=g1:
            t=(level-d0)/(d1-d0) if d1!=d0 else 0.0; R=g0+t*(g1-g0); break
    return L,R

def fmt(LR):
    L,R=LR
    if any(math.isnan(x) for x in (L,R)): return "[n/a, n/a]"
    return f"[{L:.3f}, {R:.3f}]"

def main():
    gam, chis = load_rows()
    # convert to list so we can index
    g=list(gam); c=list(chis)
    chi_min=min(c); i0=c.index(chi_min); g_best=g[i0]
    d=[x-chi_min for x in c]
    g68=ci_from_profile(g, d, 1.00)
    g95=ci_from_profile(g, d, 3.84)

    # robustness files if present
    robust = {}
    def maybe(name, fname):
        p=OUTDIR/fname
        if p.exists():
            J=json.load(open(p))
            robust[name]=float(J.get("chi2"))
    maybe("tight_scales", "joint_gamma_fix_tightscales.json")
    maybe("shared_scale", "joint_gamma_fix_sharedscale.json")
    maybe("shared_plus_tight", "joint_gamma_fix_shared_tight.json")

    # CSV table
    CSV_OUT.write_text("gamma,chi2,delta_chi2\n" + "\n".join(f"{gg:.6f},{cc:.6f},{dd:.6f}" for gg,cc,dd in zip(g,c,d)))
    # Markdown summary
    md = []
    md.append("### Joint BAO+SN: γ profile\n")
    md.append(f"- χ²_min = **{chi_min:.3f}** at **γ* = {g_best:.3f}**")
    md.append(f"- 68% CI = **{fmt(g68)}**, 95% CI = **{fmt(g95)}**  \n")
    if robust:
        md.append("**Robustness:**")
        for k,v in robust.items():
            md.append(f"- {k.replace('_',' ')}: χ² = **{v:.3f}** (Δχ² = {v-chi_min:+.3f})")
    MD_OUT.write_text("\n".join(md) + "\n")

    # LaTeX row (for a small table in manuscript)
    latex = (
        "\\newcommand{\\GammaBest}{%.3f}\n"
        "\\newcommand{\\GammaCIa}{%.3f}\n"
        "\\newcommand{\\GammaCIb}{%.3f}\n"
        "\\newcommand{\\GammaCIc}{%.3f}\n"
        "\\newcommand{\\GammaCId}{%.3f}\n"
        "%% Example row: γ* = \\GammaBest\\; (68\\%%: [\\GammaCIa,\\GammaCIb], 95\\%%: [\\GammaCIc,\\GammaCId])\n"
    ) % (g_best, g68[0], g68[1], g95[0], g95[1])
    LATEX_OUT.write_text(latex)

    print(f"best gamma = {g_best:.3f}")
    print(f"chi2_min   = {chi_min:.3f}")
    print(f"68%% CI     = {fmt(g68)}")
    print(f"95%% CI     = {fmt(g95)}")
    print(f"Wrote {CSV_OUT}\nWrote {MD_OUT}\nWrote {LATEX_OUT}")

if __name__ == "__main__":
    main()
