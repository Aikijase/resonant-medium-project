#!/usr/bin/env python3
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--families", required=True, help="Comma list, e.g., BAO_LRG,BAO_ELG,BAO_QSO,LYA,SN_LOWZ,SN_HIGHZ")
ap.add_argument("--out", required=True)
args = ap.parse_args()
rows = [{"family": f.strip(), "status": "TODO: run your pipeline with this family removed and fill Δχ², params."}
        for f in args.families.split(",")]
json.dump({"rows": rows}, open(args.out,"w"), indent=2)
print("Wrote", args.out, "families=", len(rows))
