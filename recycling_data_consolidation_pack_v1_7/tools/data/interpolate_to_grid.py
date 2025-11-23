#!/usr/bin/env python3
import argparse, csv, os, math

def read_table(path):
    rows=[]
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r or r[0].startswith("#"): continue
            rows.append(r)
    return rows[0], rows[1:]

def write_table(path, hdr, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w=csv.writer(f); w.writerow(hdr); [w.writerow(r) for r in rows]

def linear_interp(x, xs, ys):
    if x<=xs[0]: return ys[0]
    if x>=xs[-1]: return ys[-1]
    lo=0; hi=len(xs)-1
    while hi-lo>1:
        mid=(hi+lo)//2
        if xs[mid]<=x: lo=mid
        else: hi=mid
    x0,x1=xs[lo],xs[hi]; y0,y1=ys[lo],ys[hi]
    t=(x-x0)/(x1-x0)
    return y0 + t*(y1-y0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input_csv")   # must have z in col0, value in col1
    ap.add_argument("grid_csv")    # must have z in col0
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    hdr, rows = read_table(args.input_csv)
    ghdr, grows = read_table(args.grid_csv)
    xs=[]; ys=[]
    for r in rows:
        try:
            xs.append(float(r[0])); ys.append(float(r[1]))
        except: pass
    xs_sorted = sorted(zip(xs,ys))
    xs=[x for x,_ in xs_sorted]; ys=[y for _,y in xs_sorted]
    out_rows=[hdr]
    for gr in grows[1:] if grows and grows[0][0].startswith("#") else grows:
        try:
            z=float(gr[0])
        except:
            continue
        val=linear_interp(z, xs, ys) if xs else ""
        out_rows.append([z, val] + ([""]*(len(hdr)-2)))
    write_table(args.out, hdr, out_rows[1:])  # write without comment
    print("Wrote", args.out, "rows=", len(out_rows)-1)

if __name__=="__main__":
    main()
