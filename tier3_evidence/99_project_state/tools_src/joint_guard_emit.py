#!/usr/bin/env python3
import argparse, json, subprocess as sp, sys, re, os
p = argparse.ArgumentParser()
p.add_argument("--out-json", required=True)
p.add_argument("rest", nargs=argparse.REMAINDER)
args = p.parse_args()
cmd = ["python3", "joint_guard.py"] + args.rest
proc = sp.run(cmd, capture_output=True, text=True)
sys.stdout.write(proc.stdout); sys.stderr.write(proc.stderr)
m = re.search(r"chi2\s*=\s*([0-9.eE+-]+).*?A\s*=\s*([0-9.eE+-]+).*?f\s*=\s*([0-9.eE+-]+).*?phi\s*=\s*([0-9.eE+-]+).*?gamma\s*=\s*([0-9.eE+-]+)", proc.stdout, re.S)
if not m: sys.exit(0)
chi2,A,f,phi,gamma = map(float, m.groups())
out = {"chi2": chi2, "best_params": {"A": A, "f": f, "phi": phi, "gamma": gamma}}
os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
with open(args.out_json, "w") as fh: json.dump(out, fh, separators=(",", ":"), sort_keys=True)
print(f"[WRITE] {args.out_json}")
