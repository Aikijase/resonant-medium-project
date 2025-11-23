#!/usr/bin/env python3
"""
Phase-29: Recycling – Prediction Sheet vs ΛCDM

Takes your Phase-21/23/24(27) outputs and produces *falsifiable* forecasts:
- fσ8(z) deltas vs ΛCDM at z={0.5,1.0,2.0}
- w(z) proxy drift from -1 (late-time)
- A_L and ΔNeff proxies vs ΛCDM (0-offset baseline)
- A simple detectability tag per survey (DESI/Euclid/ACT/SO) using nominal 1σ bands

Inputs:
  --pred-csv    outputs/phase23/predictions.csv
  --bench-json  outputs/phase27/bench_wwi_cfg.json (preferred) or outputs/phase24/bench_wwi.json
  --source-csv  outputs/phase21/recycling_source.csv (for w_proxy trace)
  --surveys     (optional) JSON file to override default σ assumptions

Outputs:
  outputs/phase29/forecast.csv
  outputs/phase29/forecast.txt
  outputs/phase29/forecast_plot.png    (bar chart of key deltas)
"""
import argparse, os, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DEFAULT_SURVEYS = {
  # very rough nominal 1σ sensitivities (fractional or absolute) for near-term checks
  "DESI":   {"fs8_sigma": 0.02},     # ~2% per low-z bin ballpark
  "Euclid": {"fs8_sigma": 0.015},    # ~1.5%
  "ACT":    {"AL_sigma": 0.03},      # A_L absolute
  "SO":     {"AL_sigma": 0.015},     # optimistic late-time
  "CMB-S4": {"dNeff_sigma": 0.06},   # ΔNeff absolute (order-of-mag placeholder)
}

def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def main():
    ap = argparse.ArgumentParser(description="Phase-29: Recycling – Prediction Sheet vs ΛCDM")
    ap.add_argument("--pred-csv", default="outputs/phase23/predictions.csv")
    ap.add_argument("--bench-json", default="", help="Prefer P27 bench_wwi_cfg.json; fallback to P24 bench_wwi.json")
    ap.add_argument("--source-csv", default="outputs/phase21/recycling_source.csv", help="For w_proxy track")
    ap.add_argument("--surveys", default="", help="Optional JSON with {survey: {fs8_sigma, AL_sigma, dNeff_sigma}}")
    ap.add_argument("--outdir", default="outputs/phase29")
    args = ap.parse_args()

    print("=== Phase-29: Recycling – Prediction Sheet vs ΛCDM ===")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Inputs
    if not os.path.exists(args.pred_csv):
        print("[ERROR] Missing Phase-23 predictions CSV. Run tools/phase23_predict_sheet.py first.")
        print("RESULT: FAIL"); sys.exit(2)

    preds = pd.read_csv(args.pred_csv)  # z, fs8_proxy
    # Bench JSON: prefer P27 config-driven, else P24
    bench = {}
    if args.bench_json and os.path.exists(args.bench_json):
        bench = load_json(args.bench_json)
    else:
        bench = load_json("outputs/phase27/bench_wwi_cfg.json") or load_json("outputs/phase24/bench_wwi.json")

    # w_proxy trace for late-time drift estimate
    w_proxy = None
    if os.path.exists(args.source_csv):
        src = pd.read_csv(args.source_csv)
        if {"z","w_proxy"}.issubset(src.columns):
            # estimate <w+1> over low z
            low = src.sort_values("z")
            mask = low["z"]<=1.0
            if mask.any():
                w_proxy = float(np.mean(low.loc[mask,"w_proxy"].values))
            else:
                w_proxy = float(np.mean(low["w_proxy"].tail(50).values)) if len(low)>=50 else float(np.mean(low["w_proxy"]))
    # ΔNeff & A_L proxies from predictions.txt/bench JSON
    # Phase-23 wrote predictions.txt; Phase-27/24 JSON also carries proxies
    dneff = None; AL = None
    ptxt = Path(args.pred_csv).with_name("predictions.txt")
    if ptxt.exists():
        for line in open(ptxt):
            t=line.strip()
            if t.startswith("ΔNeff_proxy"): 
                try: dneff = float(t.split("=")[1]); 
                except: pass
            if t.startswith("A_L_proxy"): 
                try: AL = float(t.split("=")[1]);
                except: pass
    if dneff is None or AL is None:
        preds_json = bench.get("preds", {})
        if dneff is None: dneff = preds_json.get("dneff")
        if AL    is None: AL    = preds_json.get("al")

    # Build deltas vs ΛCDM (fs8_LCDM = 1.0; w_LCDM = -1; A_L_LCDM ~ 1 proxy shift = 0 here; ΔNeff_LCDM = 0)
    rows=[]
    for z,fs8 in zip(preds["z"].values, preds["fs8_proxy"].values):
        delta = float(fs8 - 1.0)   # deviation from ΛCDM
        rows.append({"quantity": f"fs8@z={z:g}", "delta": delta})

    # Late-time w drift (average over z<=1): w+1 deviation from 0
    if w_proxy is not None:
        rows.append({"quantity":"<w+1>@z<=1", "delta": float(w_proxy + 1.0)})

    if AL is not None:
        rows.append({"quantity":"A_L (proxy)", "delta": float(AL)})
    if dneff is not None:
        rows.append({"quantity":"ΔNeff (proxy)", "delta": float(dneff)})

    df = pd.DataFrame(rows)

    # Detectability tags
    survey_cfg = DEFAULT_SURVEYS.copy()
    if args.surveys and os.path.exists(args.surveys):
        user_cfg = load_json(args.surveys)
        for k,v in user_cfg.items():
            survey_cfg[k] = {**survey_cfg.get(k, {}), **v}

    def tag(row):
        q = row["quantity"]
        d = abs(row["delta"])
        tags=[]
        if q.startswith("fs8@z="):
            sig = survey_cfg["DESI"]["fs8_sigma"]
            if d >= 3*sig: tags.append("DESI: >3σ")
            elif d >= 2*sig: tags.append("DESI: >2σ")
            elif d >= 1*sig: tags.append("DESI: >1σ")
            sigE = survey_cfg["Euclid"]["fs8_sigma"]
            if d >= 3*sigE: tags.append("Euclid: >3σ")
            elif d >= 2*sigE: tags.append("Euclid: >2σ")
            elif d >= 1*sigE: tags.append("Euclid: >1σ")
        elif q.startswith("A_L"):
            sigA = survey_cfg["SO"]["AL_sigma"]
            if d >= 3*sigA: tags.append("SO: >3σ")
            elif d >= 2*sigA: tags.append("SO: >2σ")
            elif d >= 1*sigA: tags.append("SO: >1σ")
            sigA2 = survey_cfg["ACT"]["AL_sigma"]
            if d >= 3*sigA2: tags.append("ACT: >3σ")
            elif d >= 2*sigA2: tags.append("ACT: >2σ")
            elif d >= 1*sigA2: tags.append("ACT: >1σ")
        elif q.startswith("ΔNeff"):
            sigN = survey_cfg["CMB-S4"]["dNeff_sigma"]
            if d >= 3*sigN: tags.append("CMB-S4: >3σ")
            elif d >= 2*sigN: tags.append("CMB-S4: >2σ")
            elif d >= 1*sigN: tags.append("CMB-S4: >1σ")
        elif q.startswith("<w+1>"):
            # Map w drift ~ fs8 sensitivity (coarse): treat 0.02 as ~1σ DESI-level
            if d >= 0.06: tags.append("DESI/Euclid: >3σ (proxy)")
            elif d >= 0.04: tags.append("DESI/Euclid: >2σ (proxy)")
            elif d >= 0.02: tags.append("DESI/Euclid: >1σ (proxy)")
        return "; ".join(tags)

    df["detectability"] = df.apply(tag, axis=1)

    # Write CSV/TXT
    csv_path = outdir/"forecast.csv"
    df.to_csv(csv_path, index=False)

    txt_path = outdir/"forecast.txt"
    with open(txt_path, "w") as f:
        f.write("Phase-29: Recycling – Prediction Sheet vs ΛCDM\n")
        for _,r in df.iterrows():
            f.write(f"{r['quantity']}: Δ={r['delta']:.4g}  {r['detectability']}\n")

    # Plot: bar of absolute deltas
    png_path = outdir/"forecast_plot.png"
    plt.figure(figsize=(9,5))
    x = np.arange(len(df))
    plt.bar(x, np.abs(df["delta"].values))
    plt.xticks(x, df["quantity"].values, rotation=45, ha="right")
    plt.ylabel("|Δ vs ΛCDM|")
    plt.title("Phase-29: Recycling forecasts — magnitude of deviations")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)

    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {png_path}")
    print("\nRESULT: PASS")

if __name__ == "__main__":
    main()
