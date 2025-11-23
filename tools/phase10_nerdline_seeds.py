#!/usr/bin/env python3
# Nerdline Seeds — train-themed, seeded + parallel sweep with per-station checkpoints
import sys, subprocess, json, math, csv, pathlib, argparse, time
import numpy as np
from multiprocessing import Pool, cpu_count

LOCO = r"""
           ___
       _||__|  |  ______     🚂  TOOT! TOOT!
      (        | |      |
     /-()---() ~ ()--()
"""

def fmt_t(s):
    if s is None or s == float("inf"): return "∞"
    m, sec = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")

def bar(p, width=28): p=max(0,min(1,p)); n=int(p*width); return "█"*n+"░"*(width-n)

def run_once(args):
    py, w2, K, base_args, timeout, seed, debug_dir = args
    cmd = [py, "tools/phase10_phasecouple_demo.py"] + base_args + [
        "--omega2", str(w2), "--Kphi", str(K),
        "--prefix", f"nerd_seed{seed}_w{w2}_K{K}",
        "--seed", str(seed)
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"ret={p.returncode}")
        out = p.stdout.strip()
        # JSON sniff: take first '{' .. last '}' slice
        i = out.find('{'); j = out.rfind('}')
        if i == -1 or j == -1 or j <= i:
            raise ValueError("no JSON braces in stdout")
        payload = out[i:j+1]
        data = json.loads(payload)
        return float(data["metrics"]["sync_index"])
    except Exception as e:
        if debug_dir:
            pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)
            stem = f"w{w2}_K{K}_s{seed}"
            pathlib.Path(debug_dir, stem+".out.txt").write_text(p.stdout if 'p' in locals() else '')
            pathlib.Path(debug_dir, stem+".err.txt").write_text(p.stderr if 'p' in locals() else str(e))
        return float('nan')

def write_header(path, ks, suffix=""):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["omega2"]+[f"K={k:.5f}{suffix}" for k in ks])

def append_row(path, w2, vals):
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([f"{w2:.5f}"]+vals)

def main():
    ap = argparse.ArgumentParser(description="Phase-Coupling Express — Seeded Nerdline")
    ap.add_argument("--wmin", type=float, default=2.6)
    ap.add_argument("--wmax", type=float, default=3.0)
    ap.add_argument("--nw",   type=int,   default=25)
    ap.add_argument("--kmin", type=float, default=0.20)
    ap.add_argument("--kmax", type=float, default=0.90)
    ap.add_argument("--nk",   type=int,   default=50)
    ap.add_argument("--thr",  type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--preset", type=str, default="neuron")
    ap.add_argument("--adapt_every", type=int, default=15)
    ap.add_argument("--out_prefix", type=str, default="outputs/phase10/nerdline_seeds")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=max(1, cpu_count()//2))
    ap.add_argument("--whistle", action="store_true")
    ap.add_argument("--debug_dir", type=str, default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    base_args = ["--preset", args.preset,"--kv",str(args.kv),"--kx",str(args.kx),
                 "--eps",str(args.eps),"--adapt_every",str(args.adapt_every),
                 "--noise",str(args.noise),"--steps",str(args.steps),
                 "--burn_in",str(args.burn_in)]

    out_dir = pathlib.Path(args.out_prefix).parent; out_dir.mkdir(parents=True, exist_ok=True)
    mean_csv = pathlib.Path(f"{args.out_prefix}_mean.csv")
    std_csv  = pathlib.Path(f"{args.out_prefix}_std.csv")

    ws = np.linspace(args.wmin, args.wmax, args.nw)
    ks = np.linspace(args.kmin, args.kmax, args.nk)

    write_header(mean_csv, ks); write_header(std_csv, ks, " (std)")
    start = time.perf_counter()
    if not args.quiet:
        print(LOCO)
        print(f"🚆  Phase-Coupling Express — {args.preset.title()} Line")
        print(f"   Departure ω₂={ws[0]:.3f} → Destination ω₂={ws[-1]:.3f} | Stops={args.nw} | Cars/stop={args.nk}")
        print(f"   Seeds={args.seeds} | Parallel={args.parallel} | Thr={args.thr}\n")

    done_rows = 0
    for i, w2 in enumerate(ws):
        row_t0 = time.perf_counter()
        if not args.quiet:
            print(f"== 🛤️  Station {i+1}/{args.nw} — ω₂={w2:.3f} ==")
            if args.whistle: print("📯  Toot! Toot!")

        jobs = []
        for K in ks:
            for s in range(args.seeds):
                jobs.append((py, round(w2,3), round(K,5), base_args, args.timeout, 17+s, args.debug_dir))

        with Pool(processes=max(1, args.parallel)) as pool:
            vals = list(pool.map(run_once, jobs))

        means, stds = [], []
        idx = 0; first_lock = None
        for j, K in enumerate(ks):
            arr = [vals[idx+s] for s in range(args.seeds)]; idx += args.seeds
            arr = [v for v in arr if not math.isnan(v)]
            if arr:
                m = sum(arr)/len(arr)
                v = sum((x-m)**2 for x in arr)/len(arr)
                sdev = math.sqrt(v)
                means.append(f"{m:.6f}"); stds.append(f"{sdev:.6f}")
                if first_lock is None and m >= args.thr: first_lock=(K,m)
            else:
                means.append(""); stds.append("")
            if not args.quiet and (j % max(1, args.nk//8) == 0 or j == args.nk-1):
                prog = (i*args.nk + j + 1)/(args.nw*args.nk)
                print(f"   🚃 Car {j+1:02d}/{args.nk}: Kφ={K:.3f} → s̄={('NaN' if means[-1]=='' else f'{float(means[-1]):.3f}')}"
                      f" | Progress [{bar(prog)}] {int(100*prog)}%")

        append_row(mean_csv, w2, means); append_row(std_csv, w2, stds)

        if not args.quiet:
            row_dt = time.perf_counter()-row_t0; done_rows += 1
            remain_rows = args.nw - done_rows
            eta = row_dt * remain_rows if i>0 else float('inf')
            if first_lock: print(f"   ✅ First lock @ Kφ≈{first_lock[0]:.3f} (s̄={first_lock[1]:.3f})")
            else:          print("   ⚠️  No lock at this station.")
            print(f"   ⏱️  Row time {fmt_t(row_dt)} | ETA {fmt_t(eta)}\n")

    total_dt = time.perf_counter()-start
    print(f"Mean CSV: {mean_csv}\nStd  CSV: {std_csv}\nElapsed: {fmt_t(total_dt)}")
if __name__ == "__main__":
    main()
