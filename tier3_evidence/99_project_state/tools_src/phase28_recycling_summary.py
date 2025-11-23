#!/usr/bin/env python3
"""
Phase-28: Recycling – Summary Sheet

Collects key artifacts into a compact CSV/TXT for dashboards and handover:
  - Best params (Phase-22)
  - Predictions (Phase-23) at z={0.5,1.0,2.0}
  - WWI (Phase-24 and/or Phase-27 if present)

Outputs:
  - outputs/phase28/summary.csv
  - outputs/phase28/summary.txt
  - RESULT always PASS (pure collation)
"""
import os, json, sys
from pathlib import Path
import pandas as pd

def load_json(p: str):
    return json.load(open(p)) if os.path.exists(p) else {}

def main():
    print("=== Phase-28: Recycling – Summary Sheet ===")
    outdir = Path("outputs/phase28")
    outdir.mkdir(parents=True, exist_ok=True)

    # Best params (Phase-22)
    fit = load_json("outputs/phase22/recycling_fit.json")
    best = fit.get("best") or fit.get("best_params") or {}

    # Predictions (Phase-23)
    preds_txt = Path("outputs/phase23/predictions.txt")
    preds_csv = Path("outputs/phase23/predictions.csv")
    dneff = None
    al = None
    if preds_txt.exists():
        for line in open(preds_txt):
            line = line.strip()
            if line.startswith("ΔNeff_proxy"):
                try:
                    dneff = float(line.split("=")[1])
                except Exception:
                    pass
            if line.startswith("A_L_proxy"):
                try:
                    al = float(line.split("=")[1])
                except Exception:
                    pass

    fs8 = {}
    if preds_csv.exists():
        df = pd.read_csv(preds_csv)
        for _, r in df.iterrows():
            try:
                fs8[float(r["z"])] = float(r["fs8_proxy"])
            except Exception:
                pass

    # Scores (prefer Phase-27; else Phase-24)
    J27 = load_json("outputs/phase27/bench_wwi_cfg.json")
    J24 = load_json("outputs/phase24/bench_wwi.json")
    chosen = J27 if J27 else J24
    scores = chosen.get("scores", {})
    status = chosen.get("status", "")

    # CSV
    csv_path = outdir / "summary.csv"
    rows = [{
        "k": best.get("k"),
        "alpha": best.get("alpha"),
        "epsilon": best.get("epsilon"),
        "dNeff_proxy": dneff,
        "A_L_proxy": al,
        "fs8_z0p5": fs8.get(0.5),
        "fs8_z1p0": fs8.get(1.0),
        "fs8_z2p0": fs8.get(2.0),
        "WWI": scores.get("WWI"),
        "fs8_score": scores.get("fs8_score"),
        "A_L_score": scores.get("al_score"),
        "dNeff_score": scores.get("dneff_score"),
        "status": status,
    }]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # TXT
    txt_path = outdir / "summary.txt"
    with open(txt_path, "w") as f:
        if best:
            f.write(f"Best params: k={best.get('k')}  alpha={best.get('alpha')}  epsilon={best.get('epsilon')}\n")
        f.write(f"ΔNeff_proxy={dneff}  A_L_proxy={al}\n")
        f.write(f"fσ8_proxy: z=0.5:{fs8.get(0.5)}  z=1.0:{fs8.get(1.0)}  z=2.0:{fs8.get(2.0)}\n")
        f.write(f"WWI={scores.get('WWI')}  status={status}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print("\nRESULT: PASS")

if __name__ == "__main__":
    main()
