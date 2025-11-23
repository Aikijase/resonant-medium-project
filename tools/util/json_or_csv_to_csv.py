#!/usr/bin/env python3
import sys, json, csv, io

def flatten(d, parent=""):
    out={}
    for k,v in d.items():
        kk = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        else:
            out[kk]=v
    return out

if len(sys.argv)<3:
    print("Usage: python3 json_or_csv_to_csv.py INPUT OUTPUT.csv", file=sys.stderr); sys.exit(2)

inp, outp = sys.argv[1], sys.argv[2]
txt = open(inp, "r", encoding="utf-8").read().strip()

def write_csv(rows):
    # rows: list of dicts
    # union of keys preserves columns
    keys=[]
    seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow(r)

if txt.startswith("{") or txt.startswith("["):
    J=json.loads(txt)
    rows=None
    if isinstance(J, list):
        # list of dicts
        rows=[ flatten(x) if isinstance(x, dict) else {"value": x} for x in J ]
    elif isinstance(J, dict):
        # dict of arrays or scalars
        # find array-like fields of equal length
        lengths={}
        for k,v in J.items():
            if isinstance(v, list): lengths[k]=len(v)
        if lengths:
            # choose the modal length
            from collections import Counter
            L = Counter(lengths.values()).most_common(1)[0][0]
            cols=[k for k,v in J.items() if isinstance(v, list) and len(v)==L]
            rows=[]
            for i in range(L):
                r={}
                for k in cols: r[k]=J[k][i]
                # include scalar fields too
                for k,v in J.items():
                    if not isinstance(v, list): r[k]=v
                rows.append(r)
        else:
            # single-row dict
            rows=[ flatten(J) ]
    else:
        rows=[ {"value": J} ]
    write_csv(rows)
else:
    # already CSV, just copy
    open(outp,"w",encoding="utf-8").write(txt)

print(f"Wrote {outp}")
