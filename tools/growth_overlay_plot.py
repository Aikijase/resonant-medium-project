#!/usr/bin/env python3
import argparse, csv
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--obs", required=True)
ap.add_argument("--out", default="outputs/thrace_best/growth_overlay.png")
args = ap.parse_args()

m = read_csv(args.model); o = read_csv(args.obs)
mz = [float(r["z"]) for r in m]
mf = [float(r.get("f_sigma8", r.get("fs8", 0.0))) for r in m]
def pick(d, *keys):
    for k in keys:
        if k in d and d[k] != "":
            return d[k]
    return "nan"
oz = [float(pick(r,"z","z_eff","zeff","zmid")) for r in o]
of = [float(pick(r,"fs8","f_sigma8")) for r in o]
oe = [float(pick(r,"sigma","err","error","sigma_fs8","dfs8")) for r in o]

plt.figure()
plt.plot(mz, mf, label="Model fσ8(z)")
plt.errorbar(oz, of, yerr=oe, fmt="o", capsize=3, label="Obs")
plt.xlabel("z"); plt.ylabel("fσ8"); plt.title("Growth: model vs observations")
plt.legend(); plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Wrote {args.out}")
