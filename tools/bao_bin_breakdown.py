#!/usr/bin/env python3
import csv, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv)>1 else "outputs/thrace_best/bao_desidr1_fit_bao_residuals.csv"

# Read residual file
rows = []
with open(path, newline="") as f:
    r = csv.DictReader(f)
    cols = r.fieldnames

    # detect column names
    kind_col = None
    for c in ["kind", "metric", "type", "observable"]:
        if c in cols:
            kind_col = c
            break
    if not kind_col:
        print("ERROR: no 'kind' or 'metric' column in residual CSV")
        print("Columns found:", cols)
        sys.exit(1)

    # Standard headers used by bao_validate residuals
    zc = None
    for c in ["z", "z_eff", "zeff", "zmid"]:
        if c in cols: zc = c
    obs_c  = "obs" if "obs" in cols else "value"
    sig_c  = "sigma" if "sigma" in cols else "err"
    mod_c  = "model" if "model" in cols else None

    for d in r:
        try:
            z = float(d[zc])
            kind = d[kind_col]
            obs  = float(d[obs_c])
            sig  = float(d[sig_c])
            mod  = float(d[mod_c]) if mod_c else None
            if sig > 0:
                pull = (mod - obs) / sig
                chi2 = pull * pull
                rows.append((z, kind, chi2, pull))
        except:
            continue

# Group by DESI bin
def tracer_for_z(z, kind):
    if 0.10 <= z < 0.40: return "BGS"
    if 0.40 <= z < 0.60: return "LRG1"
    if 0.60 <= z < 0.80: return "LRG2"
    if 0.80 <= z < 1.10: return "LRG+ELG"
    if 1.10 <= z < 1.60: return "ELG"
    if kind == "DV/rd" and 0.80 <= z <= 2.10: return "QSO"
    return "OTHER"

agg = defaultdict(lambda: [0.0,0,0.0,0.0])  # chi2,sum
for z, kind, chi2, pull in rows:
    t = tracer_for_z(z, kind)
    key = f"{t}:{kind}"
    agg[key][0] += chi2
    agg[key][1] += 1
    agg[key][2] = max(agg[key][2], abs(pull))
    agg[key][3] += pull

print("block,metric,n,chi2,chi2/pt,max|pull|,sum_pull")
for k,(c2,n,mx,sp) in sorted(agg.items()):
    print(f"{k},{n},{c2:.3f},{(c2/n if n else 0):.3f},{mx:.3f},{sp:.3f}")
