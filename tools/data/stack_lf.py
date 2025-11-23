#!/usr/bin/env python3
import csv, argparse, os

def read_csv(p):
    with open(p, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_hdr = ["z","log10_Lbol_erg_s","phi_dex_Mpc3"]
    out_rows = [out_hdr]
    for path in args.inputs:
        hdr, data = read_csv(path)
        name = {n.strip(): i for i,n in enumerate(hdr)}
        zc   = name.get("z")
        Lc   = name.get("log10_Lbol_erg_s")
        pc   = name.get("phi_dex_Mpc3")
        if None in (zc,Lc,pc):
            raise SystemExit(f"{path} missing required cols; got {hdr}")
        for r in data:
            try:
                z=float(r[zc]); L=float(r[Lc]); p=float(r[pc])
            except: 
                continue
            out_rows.append([f"{z:.6f}", f"{L:.6f}", f"{p:.6e}"])

    # sort by z then L
    body = out_rows[1:]
    body.sort(key=lambda x: (float(x[0]), float(x[1])))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows([out_hdr]+body)
    print(f"Wrote {args.out} rows= {len(body)}")
if __name__ == "__main__":
    main()
