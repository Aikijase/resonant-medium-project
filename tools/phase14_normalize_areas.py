#!/usr/bin/env python3
import argparse, json, os
import pandas as pd

def domain_area(shell_csv: str) -> float:
    df = pd.read_csv(shell_csv).dropna(subset=["omega2","Kphi"])
    return float((df["omega2"].max()-df["omega2"].min()) * (df["Kphi"].max()-df["Kphi"].min()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--outdir", default="outputs/phase14")
    ap.add_argument("--prefix", default="p14_edge")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    A_dom = domain_area(args.shell_csv)

    with open(args.summary_json, "r") as f:
        rows = json.load(f)

    out = []
    for r in rows:
        if r.get("status") != "ok": 
            continue
        am = r.get("A_mask")
        at = r.get("A_trapz")
        omega2_min = r.get("omega2_min")
        omega2_max = r.get("omega2_max")
        span_w = (float(omega2_max) - float(omega2_min)) if (omega2_min is not None and omega2_max is not None) else None

        out.append({
            **r,
            "A_domain": A_dom,
            "A_mask_norm": (am / A_dom) if (am is not None and A_dom > 0) else None,
            # simple normalisation for trapz vs domain (you can change later if you prefer span-based):
            "A_trapz_norm": (at / A_dom) if (at is not None and A_dom > 0) else None,
            "omega2_span": span_w
        })

    norm_json = os.path.join(args.outdir, f"{args.prefix}_batch_summary_normalized.json")
    norm_csv  = os.path.join(args.outdir, f"{args.prefix}_batch_summary_normalized.csv")

    import csv
    keys = ["low","high","merge_mode","mean_width","A_trapz","A_mask","A_domain","A_trapz_norm","A_mask_norm",
            "omega2_min","omega2_max","omega2_span","width_csv","width_png","mask_png"]
    with open(norm_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in keys})

    with open(norm_json, "w") as f:
        json.dump(out, f, indent=2)

    print("wrote:", norm_csv)
    print("wrote:", norm_json)
    print("A_domain:", A_dom)

if __name__ == "__main__":
    main()
