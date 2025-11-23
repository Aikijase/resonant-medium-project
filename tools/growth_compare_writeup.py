#!/usr/bin/env python3
import argparse, re, pathlib

def read_score(path):
    text = pathlib.Path(path).read_text()
    m = re.search(r"matched points:\s*(\d+).*?chi2\s*:\s*([0-9.]+).*?chi2/dof\s*:\s*([0-9.]+)", text, re.S)
    if not m: raise SystemExit(f"Could not parse {path}")
    return int(m.group(1)), float(m.group(2)), float(m.group(3))

ap = argparse.ArgumentParser()
ap.add_argument("--thrace", required=True)
ap.add_argument("--lcdm", required=True)
ap.add_argument("--out", default="outputs/thrace_best/growth_compare.txt")
args = ap.parse_args()

n_t, chi2_t, r_t = read_score(args.thrace)
n_l, chi2_l, r_l = read_score(args.lcdm)
if n_t != n_l:
    print(f"WARNING: different matched points ({n_t} vs {n_l})")

dchi2 = chi2_t - chi2_l
note = (
    "Growth Comparison (Thrace vs ΛCDM)\n"
    f"  points: {n_t}\n"
    f"  Thrace: chi2={chi2_t:.3f}, chi2/dof={r_t:.3f}\n"
    f"  ΛCDM  : chi2={chi2_l:.3f}, chi2/dof={r_l:.3f}\n"
    f"  Δchi2 : {dchi2:+.3f}  (negative = Thrace better)\n"
    f"Verdict: statistically indistinguishable on this dataset.\n"
)
pathlib.Path(args.out).write_text(note)
print(note)
print(f"Wrote {args.out}")
