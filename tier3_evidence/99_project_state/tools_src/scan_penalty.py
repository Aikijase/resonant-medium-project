#!/usr/bin/env python3
import argparse, json, math
ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="phase8")
ap.add_argument("--method", choices=["sidak","bonferroni"], default="sidak")
ap.add_argument("--M", type=int, default=50, help="Number of effectively independent frequencies searched")
ap.add_argument("--p_raw", type=float, default=0.01, help="Raw p-value for observed Δχ²")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.method=="bonferroni":
    p = min(1.0, a.M * a.p_raw)
else:
    p = 1.0 - (1.0 - a.p_raw)**a.M
json.dump({"grid": a.grid, "method": a.method, "M": a.M, "p_raw": a.p_raw, "p_corrected": p}, open(a.out,"w"), indent=2)
print("Wrote", a.out, "p_corrected=", p)
