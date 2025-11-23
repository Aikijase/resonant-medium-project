#!/usr/bin/env python3
import argparse, json, os, numpy as np, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--model", default="lcdm")
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
pathlib.Path(a.outdir).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
for i in range(a.n):
    # This is a placeholder: you should replace with a call to your runner that makes a mock and fits both models.
    dchi2 = float(rng.normal(loc=0.0, scale=2.0))
    json.dump({"i": i, "delta_chi2": dchi2}, open(os.path.join(a.outdir, f"mock_{i:04d}.json"),"w"))
print("Wrote", a.n, "mock files to", a.outdir)
