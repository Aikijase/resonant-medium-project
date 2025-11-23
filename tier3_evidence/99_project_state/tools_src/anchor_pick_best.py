#!/usr/bin/env python3
import json, glob, os, shutil, sys

paths = sorted(glob.glob("outputs/anchor_*.json"))
if not paths:
    print("No outputs/anchor_*.json found.", file=sys.stderr)
    sys.exit(1)

best = None
for p in paths:
    try:
        with open(p) as f:
            J = json.load(f)
        rms = J.get("metrics", {}).get("rms")
        if rms is None:
            continue
        rms = float(rms)
        if best is None or rms < best["rms"]:
            best = {
                "path": p,
                "rms": rms,
                "fit": J.get("fit_params", {}),
                "map": J.get("mapping", J.get("map", "?")),
            }
    except Exception as e:
        print(f"Skip {p}: {e}", file=sys.stderr)

if not best:
    print("No usable JSONs.", file=sys.stderr)
    sys.exit(2)

os.makedirs("outputs", exist_ok=True)
os.makedirs("configs", exist_ok=True)

src  = os.path.abspath(best["path"])
dest = os.path.abspath("outputs/anchor_best.json")
if src != dest:
    shutil.copy2(src, dest)  # only copy if different

with open("configs/anchor_default.json", "w") as f:
    json.dump({
        "fit_params": best["fit"],
        "mapping": best["map"],
        "rms": best["rms"],
        "source_json": os.path.basename(best["path"]),
    }, f, indent=2)

print(json.dumps(best, indent=2))
