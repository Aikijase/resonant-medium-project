#!/usr/bin/env python3
import csv, re, argparse, os

def clean_name(s: str) -> str:
    t = str(s).strip().strip("[]'\"").replace('"','').replace("'",'')
    t = t.replace("\ufeff","").strip().lower()
    t = re.sub(r"\s+", "", t)
    return t

NUM = re.compile(r'^[\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')

def as_float(x):
    if x is None: return None
    t = str(x).strip()
    try: return float(t)
    except:
        m = NUM.match(t)
        return float(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.in_csv, newline="") as f:
        rows = [r for r in csv.reader(f) if r]

    if not rows: raise SystemExit("Empty input.")

    raw_hdr = rows[0]
    hdr = [clean_name(h) for h in raw_hdr]

    # map synonyms
    name = {h:i for i,h in enumerate(hdr)}
    z_col = name.get("z") or name.get("z_bin_center")
    rho_col = name.get("rho_bh_msun_mpc3") or name.get("rho_bh")

    if z_col is None or rho_col is None:
        raise SystemExit(f"Couldn't find z/rho columns in header: {raw_hdr}")

    out = [["z","rho_bh_Msun_Mpc3"]]
    for r in rows[1:]:
        # pad if short
        if len(r) <= max(z_col, rho_col):
            r = r + ['']*(max(z_col, rho_col) - len(r) + 1)
        z = as_float(r[z_col]); rho = as_float(r[rho_col])
        if z is None or rho is None: 
            continue
        out.append([f"{z:.6f}", f"{rho:.6e}"])

    # sort & dedupe by z
    body = out[1:]
    body.sort(key=lambda x: float(x[0]))
    dedup = []
    last = None
    for z,r in body:
        if z != last:
            dedup.append([z,r]); last=z
        else:
            dedup[-1] = [z,r]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows([out[0]] + dedup)
    print(f"Wrote {args.out} rows= {len(dedup)}")

if __name__ == "__main__":
    main()
