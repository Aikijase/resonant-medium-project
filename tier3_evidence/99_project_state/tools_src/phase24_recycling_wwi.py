#!/usr/bin/env python3
"""
Phase-24: Recycling – WWI Scoring + A/B vs ΛCDM

Purpose:
  - Read Phase-23 predictions (ΔNeff_proxy, A_L_proxy, fσ8 proxies).
  - Convert to a simple WWI-style score in [0,100].
  - Emit JSON, TXT, and a bar chart; print RESULT: PASS/FAIL.

Inputs:
  - --pred-csv  : outputs/phase23/predictions.csv
  - --fit-json  : outputs/phase22/recycling_fit.json (optional; echoed in reports)

Outputs:
  - outputs/phase24/bench_wwi.json
  - outputs/phase24/bench_wwi.txt
  - outputs/phase24/bench_wwi_bar.png
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_phase23(pred_csv):
    # predictions.txt carries ΔNeff_proxy and A_L_proxy lines; csv has fs8_proxy rows
    txt_path = Path(pred_csv).with_name("predictions.txt")
    dneff, al = None, None
    if txt_path.exists():
        for line in open(txt_path):
            line=line.strip()
            if line.startswith("ΔNeff_proxy"):
                try: dneff = float(line.split("=")[1])
                except: pass
            if line.startswith("A_L_proxy"):
                try: al = float(line.split("=")[1])
                except: pass
    df = pd.read_csv(pred_csv)
    z = df["z"].values.tolist()
    fs8 = df["fs8_proxy"].values.tolist()
    return {"dneff": dneff, "al": al, "z": z, "fs8": fs8}

def score_from_proxies(p):
    # fs8 component: deviation from 1.0 at z={0.5,1,2}. Smaller deviation → higher score.
    fs8_devs = [abs(1.0 - x) for x in p["fs8"]]
    mean_dev = float(np.mean(fs8_devs)) if fs8_devs else 1.0
    fs8_score = 100.0 * max(0.0, 1.0 - (mean_dev/0.10))  # 0.10 deviation → 0

    # A_L proxy: smaller variance → higher score (cap denominator to keep sane range)
    al = p["al"] if p["al"] is not None else 0.1
    al_score = 100.0 * max(0.0, 1.0 - (al/0.05))  # var 0.05 → 0

    # ΔNeff proxy: smaller → higher score
    dneff = p["dneff"] if p["dneff"] is not None else 0.2
    dneff_score = 100.0 * max(0.0, 1.0 - (dneff/0.20))  # 0.20 → 0

    # Weighted WWI (can tweak weights as you refine)
    wwi = 0.50*fs8_score + 0.25*al_score + 0.25*dneff_score
    return {
        "fs8_score": fs8_score,
        "al_score": al_score,
        "dneff_score": dneff_score,
        "WWI": min(100.0, float(wwi))
    }

def main():
    ap = argparse.ArgumentParser(description="Phase-24: Recycling – WWI Scoring + A/B vs ΛCDM")
    ap.add_argument("--pred-csv", default="outputs/phase23/predictions.csv")
    ap.add_argument("--fit-json", default="outputs/phase22/recycling_fit.json")
    ap.add_argument("--outdir", default="outputs/phase24")
    ap.add_argument("--pass-threshold", type=float, default=60.0, help="WWI >= threshold → PASS")
    args = ap.parse_args()

    print("=== Phase-24: Recycling – WWI Scoring + A/B vs ΛCDM ===")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load inputs
    if not os.path.exists(args.pred_csv):
        print("[ERROR] Missing predictions CSV from Phase-23. Run tools/phase23_predict_sheet.py first.")
        print("RESULT: FAIL"); sys.exit(2)

    preds = load_phase23(args.pred_csv)
    best = {}
    if os.path.exists(args.fit_json):
        try:
            best = json.load(open(args.fit_json)).get("best", {})
        except Exception:
            best = {}

    # Compute component scores and WWI
    scores = score_from_proxies(preds)
    status = "PASS" if scores["WWI"] >= args.pass_threshold else "FAIL"

    # LCDM reference (idealized)
    lcdm = {"fs8_score": 100.0, "al_score": 100.0, "dneff_score": 100.0, "WWI": 100.0}

    # Write JSON
    J = {
        "ok": True,
        "status": status,
        "threshold": args.pass_threshold,
        "scores": scores,
        "lcdm_ref": lcdm,
        "preds": preds,
        "best_params": best
    }
    json_path = Path(args.outdir) / "bench_wwi.json"
    with open(json_path, "w") as f: json.dump(J, f, indent=2)

    # Write TXT summary
    txt_path = Path(args.outdir) / "bench_wwi.txt"
    with open(txt_path, "w") as f:
        f.write("Phase-24: Recycling – WWI Scoring + A/B vs ΛCDM\n")
        if best:
            f.write(f"Best params (Phase-22): k={best.get('k')}  alpha={best.get('alpha')}  epsilon={best.get('epsilon')}\n")
        f.write(f"fs8_score   = {scores['fs8_score']:.2f}\n")
        f.write(f"A_L_score   = {scores['al_score']:.2f}\n")
        f.write(f"ΔNeff_score = {scores['dneff_score']:.2f}\n")
        f.write(f"WWI         = {scores['WWI']:.2f}\n")
        f.write(f"Threshold   = {args.pass_threshold:.1f}\n")
        f.write(f"RESULT: {status}\n")

    # Bar chart
    png_path = Path(args.outdir) / "bench_wwi_bar.png"
    labels = ["fs8","A_L","ΔNeff","WWI"]
    ours   = [scores["fs8_score"], scores["al_score"], scores["dneff_score"], scores["WWI"]]
    base   = [lcdm["fs8_score"],  lcdm["al_score"],  lcdm["dneff_score"],  lcdm["WWI"]]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - w/2, base, width=w, label="ΛCDM (ref)")
    plt.bar(x + w/2, ours, width=w, label="Recycling")
    plt.xticks(x, labels); plt.ylabel("Score"); plt.ylim(0, 110)
    plt.title("Phase-24: WWI components and total")
    plt.legend(); plt.tight_layout()
    plt.savefig(png_path, dpi=140)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {png_path}")
    print(f"\nRESULT: {status}")

if __name__ == "__main__":
    main()
