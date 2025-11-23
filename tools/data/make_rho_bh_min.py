#!/usr/bin/env python3
import csv, sys, argparse, re, os
num = re.compile(r'^[\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')

def as_float(s):
    if s is None: return None
    t = str(s).strip()
    try: return float(t)
    except: pass
    m = num.match(t)
    return float(m.group(1)) if m else None

ap = argparse.ArgumentParser()
ap.add_argument("in_csv")
ap.add_argument("--out", required=True)
args = ap.parse_args()

with open(args.in_csv, newline='') as f:
    rdr = csv.reader(f)
    rows = [r for r in rdr if r]

if not rows:
    sys.exit("empty input")

hdr = [h.strip() for h in rows[0]]
name = {n:i for i,n in enumerate(hdr)}
if "z" not in name or ("rho_bh_Msun_Mpc3" not in name and "rho_bh" not in name):
    sys.exit(f"Header must contain 'z' and 'rho_bh_Msun_Mpc3' (or 'rho_bh'). Got: {hdr}")

zcol = name["z"]
rhcol = name.get("rho_bh_Msun_Mpc3", name.get("rho_bh"))

out = [["z","rho_bh_Msun_Mpc3"]]
for r in rows[1:]:
    if len(r) <= max(zcol,rhcol): continue
    z = as_float(r[zcol]); rho = as_float(r[rhcol])
    if z is None or rho is None: continue
    out.append([f"{z:.6f}", f"{rho:.6e}"])

# sort & dedupe by z
body = out[1:]
body.sort(key=lambda x: float(x[0]))
dedup, last = [], None
for z,r in body:
    if z != last:
        dedup.append([z,r]); last=z
    else:
        dedup[-1] = [z,r]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline='') as f:
    csv.writer(f).writerows([out[0]] + dedup)
print(f"Wrote {args.out} rows= {len(dedup)}")
