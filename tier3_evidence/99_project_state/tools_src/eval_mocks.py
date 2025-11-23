#!/usr/bin/env python3
import argparse, json, os, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--indir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
vals=[]
for fn in sorted(os.listdir(a.indir)):
    if fn.endswith(".json"):
        try:
            vals.append(float(json.load(open(os.path.join(a.indir,fn)))["delta_chi2"]))
        except Exception: pass
import math
out = {
    "n": len(vals),
    "mean": float(np.mean(vals)) if vals else None,
    "std": float(np.std(vals, ddof=1)) if len(vals)>1 else None,
    "p_gt0": float((np.array(vals)>0).mean()) if vals else None
}
json.dump(out, open(a.out,"w"), indent=2)
print("Wrote", a.out, "n=", out["n"])
