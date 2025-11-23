#!/usr/bin/env python3
import argparse, csv, json, math, os, sys, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess

PY = sys.executable

def sync_index_at(omega2, Kphi, cfg):
    cmd = [PY, "tools/phase10_phasecouple_demo.py",
           "--preset", cfg.preset,
           "--omega2", str(omega2),
           "--kv", cfg.kv, "--kx", cfg.kx, "--Kphi", str(Kphi),
           "--eps", cfg.eps, "--adapt_every", cfg.adapt_every,
           "--noise", cfg.noise,
           "--steps", str(cfg.steps), "--burn_in", str(cfg.burn_in)]
    if cfg.seed is not None:
        cmd += ["--seed1", str(cfg.seed), "--seed2", str(cfg.seed + 101)]
    if cfg.prefix:
        cmd += ["--prefix", cfg.prefix]
    p = subprocess.run(cmd, capture_output=True, text=True)
    p.check_returncode()
    out = json.loads(p.stdout)
    return float(out["metrics"]["sync_index"])

def ensure_bracket(omega2, target, k_lo, k_hi, cfg, grow=1.5, max_k=1.5, max_tries=12):
    """Find a bracket [k_lo, k_hi] such that s(k_lo)<target and s(k_hi)>=target.
       If k_lo already ≥target, return (k_lo, slo, True). If cannot find, return (None, shi, False)."""
    slo = sync_index_at(omega2, k_lo, cfg)
    if slo >= target:
        return k_lo, slo, True
    shi = sync_index_at(omega2, k_hi, cfg)
    if shi >= target:
        return (k_lo, slo, False), (k_hi, shi, True)
    # Expand upward until we hit target or cap
    k = k_hi
    s = shi
    tries = 0
    while s < target and k < max_k and tries < max_tries:
        k = min(max_k, k * grow)
        s = sync_index_at(omega2, k, cfg)
        tries += 1
    if s >= target:
        return (k_lo, slo, False), (k, s, True)
    return None, s, False

def bisect_boundary(omega2, target, bracket, tol, max_iter, cfg):
    """Bisection inside a valid bracket ((k_lo,slo,_),(k_hi,shi,_))."""
    (k_lo, slo, _), (k_hi, shi, _) = bracket
    kL, kH = k_lo, k_hi
    sL, sH = slo, shi
    it = 0
    kmid = None; smid = None
    while it < max_iter:
        kmid = 0.5 * (kL + kH)
        smid = sync_index_at(omega2, kmid, cfg)
        if abs(smid - target) <= tol:
            return kmid, smid
        if smid >= target:
            kH, sH = kmid, smid
        else:
            kL, sL = kmid, smid
        it += 1
    return kmid, smid

def worker(task):
    """task = (omega2, seed, args_as_dict)"""
    w2, seed, A = task
    class C: pass
    cfg = C()
    cfg.preset = A["preset"]; cfg.kv = A["kv"]; cfg.kx = A["kx"]; cfg.eps = A["eps"]; cfg.noise = A["noise"]
    cfg.steps = A["steps"]; cfg.burn_in = A["burn_in"]; cfg.adapt_every = A["adapt_every"]
    cfg.seed = seed
    cfg.prefix = f"bdry_w{w2:.3f}_seed{seed}"
    target = A["target"]; tol = A["tol"]; max_iter = A["max_iter"]
    # initial bracket guess
    k_lo = A["k_lo"]; k_hi = A["k_hi"]
    try:
        # Ensure a valid bracket
        first, s, ok = ensure_bracket(w2, target, k_lo, k_hi, cfg, grow=A["grow"], max_k=A["max_k"], max_tries=A["max_grow_tries"])
        if ok is True and isinstance(first, float):  # already above target at k_lo
            return (w2, seed, first, s, target, "at_lower")
        if first is None:
            return (w2, seed, "", s, target, "unreachable")
        bracket = (first, s, ok) if isinstance(first, tuple) else first
        # Bisection
        k_est, s_est = bisect_boundary(w2, target, bracket, tol, max_iter, cfg)
        return (w2, seed, k_est, s_est, target, "ok")
    except subprocess.CalledProcessError as e:
        return (w2, seed, "", float("nan"), target, f"subprocess_error:{e.returncode}")
    except Exception as e:
        return (w2, seed, "", float("nan"), target, f"error:{type(e).__name__}")

def already_done(out_csv):
    done = set()
    if Path(out_csv).exists():
        with open(out_csv, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    w2 = float(row["omega2"]); seed = int(row["seed"])
                    done.add((round(w2,5), seed))
                except Exception:
                    continue
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w2-min", type=float, default=2.65)
    ap.add_argument("--w2-max", type=float, default=2.95)
    ap.add_argument("--w2-step", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--k-lo", type=float, default=0.10)
    ap.add_argument("--k-hi", type=float, default=0.70)
    ap.add_argument("--max-k", type=float, default=1.20)
    ap.add_argument("--grow", type=float, default=1.6)
    ap.add_argument("--max-grow-tries", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=18)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--preset", default="neuron")
    ap.add_argument("--kv", default="0.10"); ap.add_argument("--kx", default="0.20")
    ap.add_argument("--eps", default="0.06"); ap.add_argument("--noise", default="0.01")
    ap.add_argument("--adapt-every", dest="adapt_every", default="15")
    ap.add_argument("--out", default="outputs/phase10/boundary_scan_wide.csv")
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count()//2))
    args = ap.parse_args()

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)

    # Build tasks
    tasks = []
    w2 = args.w2_min
    while w2 <= args.w2_max + 1e-12:
        for seed in args.seeds:
            key = (round(w2,5), seed)
            if key not in done:
                A = dict(
                    preset=args.preset, kv=args.kv, kx=args.kx, eps=args.eps, noise=args.noise,
                    steps=args.steps, burn_in=args.burn_in, adapt_every=args.adapt_every,
                    target=args.target, tol=args.tol, max_iter=args.max_iter,
                    k_lo=args.k_lo, k_hi=args.k_hi, grow=args.grow, max_k=args.max_k, max_grow_tries=args.max_grow_tries
                )
                tasks.append((round(w2,5), seed, A))
        w2 = round(w2 + args.w2_step, 5)

    # Write header if new file
    new_file = not outp.exists()
    if new_file:
        with outp.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["omega2","seed","Kphi_min_est","sync_at_est","target","status"])

    if not tasks:
        print(f"[scan] nothing to do; all ({len(done)}) points already present.")
        return

    print(f"[scan] starting {len(tasks)} tasks with {args.jobs} workers …")
    with Pool(processes=args.jobs) as pool, outp.open("a", newline="") as f:
        wr = csv.writer(f)
        for i, res in enumerate(pool.imap_unordered(worker, tasks), 1):
            w2, seed, k_est, s_est, target, status = res
            wr.writerow([f"{w2:.5f}", seed,
                         "" if (k_est=="" or (isinstance(k_est,float) and not math.isfinite(k_est))) else f"{float(k_est):.6f}",
                         "" if (s_est=="" or (isinstance(s_est,float) and not math.isfinite(s_est))) else f"{float(s_est):.6f}",
                         f"{target:.4f}", status])
            f.flush()
            print(f"[scan] {i:04d}/{len(tasks)} w2={w2:.3f} seed={seed} -> {status}  K≈{k_est}  s≈{s_est}")

    print(f"[phase10] Wrote/updated: {outp}")

if __name__ == "__main__":
    main()
