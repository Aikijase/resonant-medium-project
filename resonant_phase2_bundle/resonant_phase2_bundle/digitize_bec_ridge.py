
#!/usr/bin/env python3
import argparse, csv, os
import matplotlib.pyplot as plt
from matplotlib.image import imread

def main():
    ap = argparse.ArgumentParser(description="Digitize BEC ridge points from a figure")
    ap.add_argument("--image", required=True, help="Path to ridge figure (png/jpg)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--omega0", type=float, default=None, help="Reference ω0 (if known). If omitted, script will prompt.")
    ap.add_argument("--label", default="BEC ridge", help="Series label")
    args = ap.parse_args()

    img = imread(args.image)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click along the ridge; press Enter when done")
    pts = plt.ginput(n=-1, timeout=0)  # unlimited until Enter
    plt.close(fig)

    if not pts:
        raise SystemExit("No points selected.")

    # Simple x->omega, y->epsilon mapping placeholders.
    # In many papers, the x-axis ~ frequency and y-axis ~ normalized amplitude.
    # If needed, rescale via known axis ticks: replace the identity maps below.
    # For now, assume image pixels map linearly onto arbitrary units; users can rescale offline.
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Normalize y so the top of the image is epsilon=1 and bottom is epsilon=0 (invert if needed).
    h = img.shape[0]
    epsilon = [(h - y)/h for y in ys]

    # Omega: set to x-pixel for now; user can apply linear calibration after exporting if desired.
    omega = xs
    omega0 = [args.omega0 if args.omega0 is not None else 1.0 for _ in xs]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega","omega0","epsilon","label"])
        for o, o0, e in zip(omega, omega0, epsilon):
            w.writerow([o, o0, e, args.label])

    print(f"Wrote {len(xs)} points to {args.out}")
    print("Tip: adjust scaling post-hoc if you have axis tick -> value mapping.")

if __name__ == "__main__":
    main()
