#!/usr/bin/env python3
import argparse, json, math, hashlib
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["SN","BAO"], required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rows=[]
for i in range(args.k):
    rows.append({"fold": i, "dataset": args.dataset, "status": "TODO: train on K-1 folds, eval on held-out; record Δχ²"})
json.dump({"k": args.k, "dataset": args.dataset, "rows": rows}, open(args.out,"w"), indent=2)
print("Wrote", args.out)
