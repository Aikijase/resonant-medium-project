#!/usr/bin/env python3
import sys, csv, argparse, os

def read_table(path):
    rows=[]
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r or r[0].startswith("#"):
                continue
            rows.append(r)
    return rows[0], rows[1:]

def write_table(path, hdr, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(hdr)
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hdr, rows = read_table(args.input_csv)
    # simple pass-through; in your environment add unit conversions & QC
    write_table(args.out, hdr, rows)
    print("Wrote", args.out, "rows=", len(rows))

if __name__ == "__main__":
    main()
