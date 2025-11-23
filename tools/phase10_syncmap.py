#!/usr/bin/env python3
import sys, subprocess, json, math, csv, pathlib, argparse, time
import numpy as np
import matplotlib.pyplot as plt

PYTHON = sys.executable

def si(w2, K, base_args, timeout):
    cmd = [PYTHON, "tools/phase10_phasecouple_demo.py"] + base_args + [
        "--omega2", str(w2), "--Kphi", str(K),
        "--prefix", f"map_w{w2}_K{K}"
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return float("nan")
        data = json.loads(p.stdout)
        return float(data["metrics"]["sync_index"])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmin", type=float, default=2.6)
    ap.add_argument("--wmax", type=float, default=3.0)
    ap.add_argument("--nw",   type=int,   default=41)
    ap.add_argument("--kmin", type=float, default=0.05)
    ap.add_argument("--kmax", type=float, default=1.0)
    ap.add_argument("--nk",   type=int,   default=40)
    ap.add_argument("--thr",  type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--preset", type=str, default="neuron")
    ap.add_argument("--adapt_every", type=int, default=15)
    ap.add_argument("--out_prefix", type=str, default="outputs/phase10/syncmap")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-sim seconds")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--probe", action="store_true", help="run one point then exit")
    args = ap.parse_args()

    base_args = [
        "--preset", args.preset,
        "--kv", str(args.kv), "--kx", str(args.kx),
        "--eps", str(args.eps),
        "--adapt_every", str(args.adapt_every),
        "--noise", str(args.noise),
        "--steps", str(args.steps),
        "--burn_in", str(args.burn_in),
    ]

    out_dir = pathlib.Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pathlib.Path(f"{args.out_prefix}.csv")
    png_path = pathlib.Path(f"{args.out_prefix}.png")

    ws = np.linspace(args.wmin, args.wmax, args.nw)
    ks = np.linspace(args.kmin, args.kmax, args.nk)
    M = np.full((args.nw, args.nk), np.nan, dtype=float)

    if args.probe:
        s = si(round(ws[len(ws)//2],3), round(ks[len(ks)//2],5), base_args, args.timeout)
        print(json.dumps({"probe": {"omega2": float(ws[len(ws)//2]),
                                    "Kphi": float(ks[len(ks)//2]),
                                    "sync_index": None if math.isnan(s) else s}}, indent=2))
        return

    total = args.nw * args.nk
    start = time.perf_counter()
    done = 0
    try:
        for i, w2 in enumerate(ws):
            row_start = time.perf_counter()
            for j, K in enumerate(ks):
                M[i, j] = si(round(w2,3), round(K,5), base_args, args.timeout)
                done += 1
                if args.verbose and (done % 5 == 0 or j == args.nk-1):
                    elapsed = time.perf_counter() - start
                    rate = done / max(elapsed, 1e-6)
                    remain = (total - done) / max(rate, 1e-9)
                    print(f"[{done:4d}/{total}] w2={w2:.3f} k={K:.3f} "
                          f"s={M[i,j] if not math.isnan(M[i,j]) else 'NaN'} "
                          f"| {rate:.1f} it/s ETA {remain/60:.1f} min",
                          flush=True)
            if args.verbose:
                print(f" row {i+1}/{args.nw} took {time.perf_counter()-row_start:.1f}s", flush=True)
    except KeyboardInterrupt:
        print("\n# Interrupted — saving partial results.", file=sys.stderr)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega2"] + [f"K={k:.5f}" for k in ks])
        for i, w2 in enumerate(ws):
            w.writerow([f"{w2:.5f}"] + [("" if math.isnan(x) else f"{x:.6f}") for x in M[i]])

    try:
        v = np.clip(M, args.thr, 1.0)
        plt.figure()
        plt.imshow(v.T, origin="lower", aspect="auto",
                   extent=[ws[0], ws[-1], ks[0], ks[-1]])
        plt.colorbar(label="sync_index (clipped ≥ threshold)")
        plt.xlabel(r"$\omega_2$")
        plt.ylabel(r"$K_\phi$")
        plt.title("Phase-coupling sync map")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        print(f"CSV: {csv_path}")
        print(f"PNG: {png_path}")
    except Exception as e:
        print(f"CSV: {csv_path}")
        print(f"(plot skipped: {e})")

if __name__ == "__main__":
    main()
