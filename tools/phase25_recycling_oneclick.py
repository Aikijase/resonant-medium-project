#!/usr/bin/env python3
"""
Phase-25: Recycling – One-Click Orchestrator

Runs:
  1) Phase-21 with best params from Phase-22
  2) Phase-23 predictions
  3) Phase-24 WWI scoring

Writes:
  - outputs/phase25/oneclick_summary.txt
  - RESULT: PASS/FAIL (mirrors Phase-24’s status)
"""
import json, os, sys, subprocess as sp
from pathlib import Path
from datetime import datetime

def run(cmd):
    print("+", " ".join(cmd))
    p = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    print(p.stdout)
    return p.returncode, p.stdout

def main():
    print("=== Phase-25: Recycling – One-Click Orchestrator ===")
    outdir = Path("outputs/phase25"); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load best params from Phase-22
    fit_path = Path("outputs/phase22/recycling_fit.json")
    if not fit_path.exists():
        print("[ERROR] Missing outputs/phase22/recycling_fit.json. Run Phase-22 first.")
        print("RESULT: FAIL"); sys.exit(2)
    J = json.load(open(fit_path))
    best = J.get("best") or J.get("best_params") or {}
    k = str(best.get("k", 0.12))
    alpha = str(best.get("alpha", 1.1))
    eps = str(best.get("epsilon", 0.7))

    # 2) Phase-21
    rc, _ = run(["python3", "tools/phase21_recycling_ode.py",
                 "--k", k, "--alpha", alpha, "--epsilon", eps])
    if rc != 0:
        print("RESULT: FAIL"); sys.exit(2)

    # 3) Phase-23
    rc, _ = run(["python3", "tools/phase23_predict_sheet.py"])
    if rc != 0:
        print("RESULT: FAIL"); sys.exit(2)

    # 4) Phase-24
    rc, _ = run(["python3", "tools/phase24_recycling_wwi.py"])
    if rc != 0:
        print("RESULT: FAIL"); sys.exit(2)

    # 5) Summarize
    bench = json.load(open("outputs/phase24/bench_wwi.json"))
    status = bench.get("status","PASS")
    scores = bench.get("scores",{})
    preds = bench.get("preds",{})
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    summary = []
    summary.append("# Recycling One-Click Summary (Phase-25)")
    summary.append(f"Timestamp: {ts}")
    if best: summary.append(f"Best params: k={k}, alpha={alpha}, epsilon={eps}")
    summary.append("")
    summary.append("WWI components:")
    summary.append(f"  fs8_score   = {scores.get('fs8_score',float('nan')):.2f}")
    summary.append(f"  A_L_score   = {scores.get('al_score',float('nan')):.2f}")
    summary.append(f"  ΔNeff_score = {scores.get('dneff_score',float('nan')):.2f}")
    summary.append(f"  WWI         = {scores.get('WWI',float('nan')):.2f}")
    summary.append("")
    if preds:
        z = preds.get("z",[]); fs8 = preds.get("fs8",[])
        for zi, gi in zip(z, fs8):
            summary.append(f"  fσ8_proxy(z={zi}) = {gi:.6f}")
    summary.append("")
    summary.append(f"RESULT: {status}")

    out = outdir / "oneclick_summary.txt"
    out.write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote {out}\n")
    print(f"RESULT: {status}")

if __name__ == "__main__":
    main()
