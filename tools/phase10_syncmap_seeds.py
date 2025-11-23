#!/usr/bin/env python3
import sys, subprocess, json, math, csv, pathlib, argparse, time, os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

PYTHON = sys.executable

def git_hash():
    try:
        h = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
        return h
    except Exception:
        return None

def run_once(args_tuple):
    (w2, K, base_args, timeout, seed) = args_tuple
    cmd = [PYTHON, "tools/phase10_phasecouple_demo.py"] + base_args + [
        "--omega2", str(w2), "--Kphi", str(K),
        "--prefix", f"seed{seed}_w{w2}_K{K}",
        "--seed", str(seed)
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return float("nan")
        data = json.loads(p.stdout)
        return float(data["metrics"]["sync_index"])
    except Exception:
        return float("nan")

def render_map(png_path, ws, ks, Mmean, thr):
    v = np.clip(Mmean, thr, 1.0)
    plt.figure()
    plt.imshow(v.T, origin="lower", aspect="auto",
               extent=[ws[0], ws[-1], ks[0], ks[-1]])
    plt.colorbar(label="sync_index mean (clipped ≥ threshold)")
    plt.xlabel(r"$\omega_2$")
    plt.ylabel(r"$K_\phi$")
    plt.title("Phase-coupling sync map (mean over seeds)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)

def write_csv(path, ws, ks, M, label_fmt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega2"] + [label_fmt(k) for k in ks])
        for i, w2 in enumerate(ws):
            row = [f"{w2:.5f}"]
            for val in M[i]:
                row.append("" if (val is None or math.isnan(val)) else f"{val:.6f}")
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Seeded, parallel sync heatmap with provenance")
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
    ap.add_argument("--out_prefix", type=str, default="outputs/phase10/syncmap_seeds")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-sim seconds")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=max(1, cpu_count()//2))
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
    csv_mean = pathlib.Path(f"{args.out_prefix}_mean.csv")
    csv_std  = pathlib.Path(f"{args.out_prefix}_std.csv")
    png_map  = pathlib.Path(f"{args.out_prefix}.png")
    prov_js  = pathlib.Path(f"{args.out_prefix}_provenance.json")

    ws = np.linspace(args.wmin, args.wmax, args.nw)
    ks = np.linspace(args.kmin, args.kmax, args.nk)

    # Build all jobs
    jobs = []
    for i, w2 in enumerate(ws):
        for j, K in enumerate(ks):
            for s in range(args.seeds):
                jobs.append((round(w2,3), round(K,5), base_args, args.timeout, 17+s))  # deterministic seeds

    # Run in parallel
    Mmean = np.full((args.nw, args.nk), np.nan)
    Mstd  = np.full((args.nw, args.nk), np.nan)

    t0 = time.perf_counter()
    with Pool(processes=max(1, args.parallel)) as pool:
        results = pool.map(run_once, jobs)

    # Fold results into mean/std
    idx = 0
    for i in range(args.nw):
        for j in range(args.nk):
            vals = []
            for s in range(args.seeds):
                vals.append(results[idx]); idx += 1
            arr = np.array([v for v in vals if not math.isnan(v)], dtype=float)
            if arr.size > 0:
                Mmean[i, j] = float(np.mean(arr))
                Mstd[i, j]  = float(np.std(arr, ddof=0))

    # Save artifacts
    write_csv(csv_mean, ws, ks, Mmean, lambda k: f"K={k:.5f}")
    write_csv(csv_std , ws, ks, Mstd , lambda k: f"K={k:.5f} (std)")
    render_map(png_map, ws, ks, Mmean, args.thr)

    prov = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "git_hash": git_hash(),
        "runner": "phase10_syncmap_seeds.py",
        "params": {
            "wmin": args.wmin, "wmax": args.wmax, "nw": args.nw,
            "kmin": args.kmin, "kmax": args.kmax, "nk": args.nk,
            "thr": args.thr, "steps": args.steps, "burn_in": args.burn_in,
            "noise": args.noise, "eps": args.eps, "kv": args.kv, "kx": args.kx,
            "preset": args.preset, "adapt_every": args.adapt_every,
            "timeout": args.timeout, "seeds": args.seeds, "parallel": args.parallel
        },
        "outputs": {
            "csv_mean": str(csv_mean), "csv_std": str(csv_std), "png_map": str(png_map)
        },
        "notes": "Mean/std across seeds; threshold used only for plotting (clipping)."
    }
    prov_js.write_text(json.dumps(prov, indent=2))

    elapsed = time.perf_counter() - t0
    print(f"Mean CSV: {csv_mean}")
    print(f"Std  CSV: {csv_std}")
    print(f"PNG:      {png_map}")
    print(f"Provenance: {prov_js}")
    print(f"Elapsed: {elapsed:.1f}s  (parallel={args.parallel}, seeds={args.seeds})")

if __name__ == "__main__":
    main()
