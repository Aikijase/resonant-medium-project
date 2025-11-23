#!/usr/bin/env python3
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--inflate", default="0.9,1.0,1.1,1.25")
ap.add_argument("--prune-offdiag", default="0,1")
ap.add_argument("--out", required=True)
args = ap.parse_args()
infl = [float(x) for x in args.inflate.split(",")]
prunes = [int(x) for x in args.prune_offdiag.split(",")]
rows = []
for a in infl:
    for p in prunes:
        rows.append({"inflate": a, "prune_offdiag": bool(p), "instruction": "Re-run your fit with these covariance tweaks and record Δχ²/params."})
with open(args.out, "w") as f:
    json.dump({"n": len(rows), "rows": rows}, f, indent=2)
print("Wrote", args.out, "rows=", len(rows))
