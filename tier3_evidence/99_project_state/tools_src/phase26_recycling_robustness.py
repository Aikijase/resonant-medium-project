#!/usr/bin/env python3
"""
Phase-26: Recycling – Robustness Sweep

Runs N jittered realizations around Phase-22 best params (and optional BH curve jitter),
pipes each through Phase-21 → Phase-23 → Phase-24, and measures WWI stability.

Outputs:
  - outputs/phase26/robustness.csv
  - outputs/phase26/robustness_report.txt
  - outputs/phase26/robustness_hist.png
  - RESULT: PASS/FAIL  (PASS if WWI mean >= pass_threshold and std <= std_threshold)
"""
import argparse, os, sys, json, subprocess as sp
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run(cmd):
    p = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    return p.returncode, p.stdout

def main():
    ap = argparse.ArgumentParser(description="Phase-26: Robustness Sweep")
    ap.add_argument("--n", type=int, default=25, help="Number of jitter draws")
    ap.add_argument("--k-sigma-frac", type=float, default=0.25, help="Rel sigma for k")
    ap.add_argument("--alpha-sigma", type=float, default=0.15, help="Abs sigma for alpha")
    ap.add_argument("--eps-sigma", type=float, default=0.15, help="Abs sigma for epsilon")
    ap.add_argument("--pass-threshold", type=float, default=60.0)
    ap.add_argument("--std-threshold", type=float, default=5.0)
    ap.add_argument("--outdir", default="outputs/phase26")
    args = ap.parse_args()

    print("=== Phase-26: Recycling – Robustness Sweep ===")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load best params
    fit = json.load(open("outputs/phase22/recycling_fit.json"))
    best = fit.get("best") or fit.get("best_params") or {}
    k0 = float(best.get("k", 0.12))
    a0 = float(best.get("alpha", 1.1))
    e0 = float(best.get("epsilon", 0.7))

    rows = []
    for i in range(args.n):
        # Jitter params
        k = max(1e-6, np.random.normal(k0, max(1e-9, args.k_sigma_frac*abs(k0))))
        a = np.random.normal(a0, args.alpha_sigma)
        e = np.clip(np.random.normal(e0, args.eps_sigma), 0.0, 2.0)

        # Run pipeline
        rc,_ = run(["python3","tools/phase21_recycling_ode.py","--k",str(k),"--alpha",str(a),"--epsilon",str(e)])
        if rc!=0: rows.append({"i":i,"k":k,"alpha":a,"epsilon":e,"wwi":np.nan,"status":"FAIL"}); continue
        rc,_ = run(["python3","tools/phase23_predict_sheet.py"])
        if rc!=0: rows.append({"i":i,"k":k,"alpha":a,"epsilon":e,"wwi":np.nan,"status":"FAIL"}); continue
        rc,_ = run(["python3","tools/phase24_recycling_wwi.py","--pass-threshold",str(args.pass_threshold)])
        if rc!=0: rows.append({"i":i,"k":k,"alpha":a,"epsilon":e,"wwi":np.nan,"status":"FAIL"}); continue

        J=json.load(open("outputs/phase24/bench_wwi.json"))
        wwi=float(J.get("scores",{}).get("WWI",np.nan))
        status=J.get("status","PASS")
        rows.append({"i":i,"k":k,"alpha":a,"epsilon":e,"wwi":wwi,"status":status})

    df=pd.DataFrame(rows)
    csv_path=outdir/"robustness.csv"; df.to_csv(csv_path,index=False)

    # Stats
    w = df["wwi"].dropna().values
    mean = float(np.nanmean(w)) if w.size else float("nan")
    std  = float(np.nanstd(w))  if w.size else float("nan")
    passes = int((df["status"]=="PASS").sum())
    total  = len(df)

    # Report
    txt_path=outdir/"robustness_report.txt"
    with open(txt_path,"w") as f:
        f.write("Phase-26: Recycling – Robustness Sweep\n")
        f.write(f"Draws: {total}\n")
        f.write(f"PASS count: {passes}\n")
        f.write(f"WWI mean: {mean:.3f}\n")
        f.write(f"WWI std : {std:.3f}\n")
        f.write(f"Thresholds: pass>={args.pass_threshold:.1f}, std<={args.std_threshold:.1f}\n")

    # Plot
    png_path=outdir/"robustness_hist.png"
    plt.figure(figsize=(8,5))
    plt.hist(df["wwi"].dropna().values, bins=12)
    plt.axvline(args.pass_threshold,color="k",linestyle="--")
    plt.xlabel("WWI"); plt.ylabel("Count"); plt.title("Phase-26: WWI robustness")
    plt.tight_layout(); plt.savefig(png_path,dpi=140)

    ok = (mean>=args.pass_threshold) and (std<=args.std_threshold) if w.size else False
    print(f"Wrote {csv_path}"); print(f"Wrote {txt_path}"); print(f"Wrote {png_path}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")

if __name__=="__main__":
    main()
