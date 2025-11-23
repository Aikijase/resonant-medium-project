#!/usr/bin/env python3
import argparse, csv, os
import matplotlib.pyplot as plt
from matplotlib.image import imread
def main():
    ap = argparse.ArgumentParser(description="Digitize BEC ridge points from a figure")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--omega0", type=float, default=None)
    ap.add_argument("--label", default="BEC ridge")
    args = ap.parse_args()
    img = imread(args.image)
    fig, ax = plt.subplots(); ax.imshow(img)
    ax.set_title("Click along the ridge; press Enter when done")
    pts = plt.ginput(n=-1, timeout=0); plt.close(fig)
    if not pts: raise SystemExit("No points selected.")
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; h = img.shape[0]
    epsilon = [(h - y)/h for y in ys]  # top=1 bottom=0
    omega = xs; omega0 = [args.omega0 if args.omega0 is not None else 1.0 for _ in xs]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w = csv.writer(f); w.writerow(["omega","omega0","epsilon","label"])
        for o,o0,e in zip(omega,omega0,epsilon): w.writerow([o,o0,e,args.label])
    print(f"Wrote {len(xs)} points to {args.out}")
if __name__ == "__main__": main()
