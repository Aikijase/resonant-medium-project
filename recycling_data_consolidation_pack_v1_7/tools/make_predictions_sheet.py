#!/usr/bin/env python3
import argparse, csv, os, datetime, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", default=".")
    ap.add_argument("--out", default="docs/predictions_sheet.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Predictions Sheet — generated " + datetime.datetime.now().isoformat(timespec="seconds")])
        w.writerow(["id","observable","z_range","direction","magnitude_hint","dataset_needed","status","notes"])
        w.writerow(["CR-001","Δw_eff(z) low-freq mode","0–2","oscillatory","~few %","eos_w_z","pending",""])
        w.writerow(["CR-002","g×κ (galaxy–CMB lensing) boost","0.3–1.0","positive","tbd","external cross-correlations","pending","Phase-4 follow-on"])
    open(os.path.join(args.pack_root,"docs","predictions_sheet.README"),"w").write(
        "Add new rows as predictions are made. Keep IDs stable for publication.")
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
