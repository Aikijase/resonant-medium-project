#!/usr/bin/env python3
import argparse, subprocess, sys, json, csv
from statistics import mean
from pathlib import Path
import re as _re

def seed_supported(py):
    try:
        h = subprocess.run([py, 'tools/phase10_phasecouple_demo.py', '--help'],
                           capture_output=True, text=True, timeout=10)
        hh = (h.stdout or '') + (h.stderr or '')
        return ('--seed' in hh)
    except Exception:
        return False

def run_demo(py, w2, K, steps, burn_in, noise, kv, kx, eps, adapt_every,
             seed, timeout, allow_seed, verbose, debug_dir):
    cmd = [py, 'tools/phase10_phasecouple_demo.py',
           '--preset','neuron', '--omega2',str(w2), '--Kphi',str(K),
           '--kv',str(kv), '--kx',str(kx), '--eps',str(eps), '--adapt_every',str(adapt_every),
           '--steps',str(steps), '--burn_in',str(burn_in), '--noise',str(noise),
           '--prefix', f'probe_w{w2}_K{K}_s{seed}']
    if allow_seed:
        cmd += ['--seed', str(seed)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            if verbose:
                print(f"    · ret={p.returncode} | stderr: {(p.stderr or '').strip()[:120]}")
            if debug_dir:
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                stem = f"w{w2}_K{K}_s{seed}"
                Path(debug_dir, stem+".cmd").write_text(' '.join(cmd))
                Path(debug_dir, stem+".out.txt").write_text(p.stdout or '')
                Path(debug_dir, stem+".err.txt").write_text(p.stderr or '')
            return None
        out = p.stdout or ''
        m = _re.search(r'\{.*\}', out, flags=_re.S)
        if not m:
            if verbose:
                print("    · no-json | first stdout chars:", repr(out[:120]))
            if debug_dir:
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                stem = f"w{w2}_K{K}_s{seed}"
                Path(debug_dir, stem+".cmd").write_text(' '.join(cmd))
                Path(debug_dir, stem+".out.txt").write_text(out)
                Path(debug_dir, stem+".err.txt").write_text(p.stderr or '')
            return None
        payload = m.group(0)
        return float(json.loads(payload)["metrics"]["sync_index"])
    except Exception as e:
        if verbose:
            print("    · exception:", type(e).__name__, str(e)[:120])
        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            stem = f"w{w2}_K{K}_s{seed}"
            Path(debug_dir, stem+".cmd").write_text(' '.join(cmd))
        return None

def mean_sync(py, w2, K, steps, burn_in, noise, kv, kx, eps, adapt_every,
             seeds, timeout, allow_seed, verbose, debug_dir):
    vals = [run_demo(py, w2, K, steps, burn_in, noise, kv, kx, eps, adapt_every,
                     s, timeout, allow_seed, verbose, debug_dir) for s in seeds]
    vals = [v for v in vals if v is not None]
    return (mean(vals) if vals else None), len(vals)

def find_k_for_w(py, w2, thr, kmin, kmax, dk, kcap,
                 steps, burn_in, noise, kv, kx, eps, adapt_every,
                 seeds, timeout, tol, allow_seed, verbose, debug_dir,
                 max_bisect=20):
    # Ramp K upward, auto-expand ceiling if needed
    K = kmin
    s, _ = mean_sync(py, w2, K, steps, burn_in, noise, kv, kx, eps, adapt_every,
                     seeds, timeout, allow_seed, verbose, debug_dir)
    if verbose: print(f"  start Kφ={K:.3f} s̄={s}")
    lastK, lastS = K, s
    while True:
        if s is not None and s >= thr:
            # bracket found
            klo = lastK if (lastS is not None and lastS < thr) else max(kmin, K - dk)
            khi = K
            break
        K += dk
        if K > kmax:
            if kmax >= kcap:
                return None
            kmax = min(kcap, kmax*1.25 + 0.05)
            if verbose: print(f"  expand kmax → {kmax:.3f}")
        lastK, lastS = K, s
        s, _ = mean_sync(py, w2, K, steps, burn_in, noise, kv, kx, eps, adapt_every,
                         seeds, timeout, allow_seed, verbose, debug_dir)
        if verbose: print(f"  probe Kφ={K:.3f} s̄={s}")

    # Bisection refine
    slo, _ = mean_sync(py, w2, klo, steps, burn_in, noise, kv, kx, eps, adapt_every,
                       seeds, timeout, allow_seed, verbose, debug_dir)
    if slo is None: slo = thr - 1e-3
    for _ in range(max_bisect):
        km = 0.5*(klo + khi)
        sm, _ = mean_sync(py, w2, km, steps, burn_in, noise, kv, kx, eps, adapt_every,
                          seeds, timeout, allow_seed, verbose, debug_dir)
        if sm is None:
            klo = km
            continue
        if sm >= thr: khi = km
        else: klo = km
        if abs(khi - klo) < tol: break
    return khi

def main():
    ap = argparse.ArgumentParser(description="Seeded boundary scanner with auto-expand + bisection")
    ap.add_argument("--wmin", type=float, required=True)
    ap.add_argument("--wmax", type=float, required=True)
    ap.add_argument("--nw", type=int, required=True)
    ap.add_argument("--kmin", type=float, required=True)
    ap.add_argument("--kmax", type=float, required=True)
    ap.add_argument("--dk", type=float, default=0.05)
    ap.add_argument("--kcap", type=float, default=1.8)
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--adapt_every", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--debug_dir", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    ws = [args.wmin + i*(args.wmax-args.wmin)/(args.nw-1) for i in range(args.nw)]
    seed_list = [17+i for i in range(args.seeds)]
    allow_seed = False  # forced off: child expects --seed1/--seed2 only

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["omega2","Kphi_min_at_thr"])
        for i, w2 in enumerate(ws, 1):
            if args.verbose: print(f"[{i}/{len(ws)}] ω₂={w2:.3f}")
            k = find_k_for_w(py, w2, args.thr, args.kmin, args.kmax, args.dk, args.kcap,
                             args.steps, args.burn_in, args.noise, args.kv, args.kx, args.eps, args.adapt_every,
                             seed_list, args.timeout, args.tol, allow_seed, args.verbose, args.debug_dir)
            w.writerow([f"{w2:.5f}", "NA" if k is None else f"{k:.6f}"])
            if args.verbose:
                print("   →", ("NA (no lock)" if k is None else f"Kφ_min≈{k:.3f}"))
    print(f"CSV: {args.out_csv}")

if __name__ == "__main__":
    main()
