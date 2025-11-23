#!/usr/bin/env python3
import csv, re, argparse, os, ast

NUM = re.compile(r'^[\s]*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?).*')

def as_float(s):
    if s is None: return None
    t = str(s).strip()
    try: return float(t)
    except: pass
    m = NUM.match(t)
    if m:
        try: return float(m.group(1))
        except: return None
    return None

def read_rows(path):
    txt = open(path, 'r', newline='').read()
    # Detect delimiter
    delim = ',' if txt.count(',') >= txt.count('\t') else '\t'
    rows = list(csv.reader(txt.splitlines(), delimiter=delim))
    # Repair single-cell "list-like" header if present
    if rows and len(rows[0]) == 1 and rows[0][0].strip().startswith('['):
        try:
            hdr = [h.strip() for h in ast.literal_eval(rows[0][0])]
        except Exception:
            hdr = rows[0]
        data = []
        for r in rows[1:]:
            if len(r) == 1 and r[0].strip().startswith('['):
                try:
                    data.append([c for c in ast.literal_eval(r[0])])
                except Exception:
                    pass
            else:
                data.append(r)
        return hdr, data
    else:
        return (rows[0], rows[1:]) if rows else ([], [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hdr, data = read_rows(args.in_csv)
    name = {n.strip(): i for i,n in enumerate(hdr)}

    # Likely column names
    z_col = name.get("z") or name.get("z_bin_center")
    rho_col = name.get("rho_bh_Msun_Mpc3") or name.get("rho_bh")
    if z_col is None or rho_col is None:
        raise SystemExit(f"Couldn't find z/rho columns in header: {hdr}")

    out = [["z","rho_bh_Msun_Mpc3"]]
    for r in data:
        if not r: continue
        # pad short rows
        if len(r) <= max(z_col, rho_col):
            r = r + ['']*(max(z_col, rho_col)-len(r)+1)
        z = as_float(r[z_col]); rho = as_float(r[rho_col])
        if z is None or rho is None: 
            continue
        out.append([f"{z:.6f}", f"{rho:.6e}"])

    # sort + dedupe
    body = out[1:]
    body.sort(key=lambda x: float(x[0]))
    dedup, seen = [], None
    for z,r in body:
        if z != seen:
            dedup.append([z,r]); seen = z
        else:
            dedup[-1] = [z,r]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows([out[0]] + dedup)
    print(f"Wrote {args.out} rows= {len(dedup)}")
if __name__ == "__main__":
    main()
