#!/usr/bin/env python3
import argparse, csv, math, sys
from pathlib import Path

def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return [{k: (float(v) if k in ("z","D","f","f_sigma8","fs8","sigma","err","obs") and v else v)
                 for k,v in row.items()} for row in r]

def nearest_match(model, z, tol=0.003):
    # binary search not needed; model is dense. pick nearest within tol
    best, dz = None, 1e9
    for m in model:
        d = abs(m["z"] - z)
        if d < dz:
            best, dz = m, d
    return best if dz <= tol else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CSV with columns: z,D,f,f_sigma8")
    ap.add_argument("--obs", required=True, help="CSV with columns: z,fs8,sigma (or err)")
    ap.add_argument("--out-prefix", default="outputs/thrace_best/growth_score")
    ap.add_argument("--tol", type=float, default=0.003, help="z matching tolerance")
    args = ap.parse_args()

    model = read_csv(args.model)
    obs = read_csv(args.obs)
    for o in obs:
        if "sigma" not in o or o["sigma"] in ("", None):
            o["sigma"] = o.get("err", None)
        if o["sigma"] is None:
            print("ERROR: obs rows need sigma/err", file=sys.stderr); sys.exit(2)

    rows = []
    chi2 = 0.0
    for o in obs:
        m = nearest_match(model, o["z"], tol=args.tol)
        if not m: continue
        resid = m["f_sigma8"] - o["fs8"]
        w = (resid / o["sigma"])**2
        chi2 += w
        rows.append({
            "z": o["z"],
            "fs8_obs": o["fs8"],
            "sigma": o["sigma"],
            "fs8_model": m["f_sigma8"],
            "resid": resid,
            "pull": resid / o["sigma"]
        })

    Path(Path(args.out_prefix).parent).mkdir(parents=True, exist_ok=True)
    # write residuals table
    out_csv = f"{args.out_prefix}_residuals.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z","fs8_obs","sigma","fs8_model","resid","pull"])
        w.writeheader(); w.writerows(rows)

    # scorecard
    n = len(rows)
    dof = max(n - 0, 1)  # no free-parameter count applied here; keep it simple
    with open(f"{args.out_prefix}_scorecard.txt", "w") as f:
        f.write(f"Growth Scorecard\n")
        f.write(f" matched points: {n}\n")
        f.write(f" chi2          : {chi2:.3f}\n")
        f.write(f" chi2/dof      : {chi2/max(dof,1):.3f}\n")
        if n == 0:
            f.write(" WARNING: 0 points matched. Try a bigger --tol.\n")

if __name__ == "__main__":
    main()
