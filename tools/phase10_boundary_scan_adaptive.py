#!/usr/bin/env python3
import sys, subprocess, json, math, csv, pathlib, argparse

PYTHON = sys.executable

def si(w2, K, base_args):
    cmd = [PYTHON, "tools/phase10_phasecouple_demo.py"] + base_args + [
        "--omega2", str(w2), "--Kphi", str(K),
        "--prefix", f"bdry_w{w2}_K{K}"
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # Return NaN so the caller can decide how to handle
        return float("nan")
    try:
        data = json.loads(p.stdout)
        return float(data["metrics"]["sync_index"])
    except Exception:
        return float("nan")

def reaches_threshold(w2, K, base_args, thr):
    s = si(w2, K, base_args)
    return (not math.isnan(s)) and (s >= thr), s

def find_Kmin(w2, base_args, thr, kmin, kmax, n_bracket=10, tol=1e-3, max_it=20):
    """
    Find minimal K in [kmin,kmax] that reaches s>=thr.
    Returns (Kmin, s_at_Kmin) or (None, None) if not found.
    """
    # 1) Bracket: coarse scan to find first hitting bin
    Ks = [kmin + i*(kmax-kmin)/n_bracket for i in range(n_bracket+1)]
    hit_idx = None
    last_s = float("nan")
    for i, K in enumerate(Ks):
        ok, s = reaches_threshold(w2, K, base_args, thr)
        last_s = s
        if ok:
            hit_idx = i
            break
    if hit_idx is None:
        return None, None  # never reaches threshold up to kmax

    # 2) Refine with binary search between previous bin and this bin
    a = Ks[max(hit_idx-1, 0)]
    b = Ks[hit_idx]
    # Ensure a is below threshold, b is above (or equal)
    # If hit_idx==0, a==b; back off a bit
    if a == b:
        a = max(kmin, b - (kmax-kmin)/n_bracket)

    # Safety checks
    for _ in range(max_it):
        m = 0.5*(a+b)
        ok_m, s_m = reaches_threshold(w2, m, base_args, thr)
        if ok_m:
            b = m
        else:
            a = m
        if abs(b-a) <= tol*max(1.0, b):
            # Evaluate at b (the first reaching point)
            ok_b, s_b = reaches_threshold(w2, b, base_args, thr)
            return (b, s_b if ok_b else s_m)
    # Fallback after iterations
    ok_b, s_b = reaches_threshold(w2, b, base_args, thr)
    return (b, s_b if ok_b else last_s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", type=float, default=2.8)
    ap.add_argument("--deltas", type=str,
                    default="-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,-0.02,0.0,0.02,0.04,0.06,0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--kmin", type=float, default=0.05)
    ap.add_argument("--kmax", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--preset", type=str, default="neuron")
    ap.add_argument("--adapt_every", type=int, default=15)
    ap.add_argument("--out_prefix", type=str, default="outputs/phase10/boundary_adaptive")
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    w2_list = [round(args.w1 + d, 2) for d in deltas]

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
    json_path = pathlib.Path(f"{args.out_prefix}.json")

    rows = []
    print("delta_omega,omega2,Kphi_min,sync_index")
    for w2 in w2_list:
        Kmin, s_at = find_Kmin(
            w2, base_args, args.thr, args.kmin, args.kmax,
            n_bracket=16, tol=1e-3, max_it=24
        )
        if Kmin is None:
            print(f"{w2 - args.w1:.3f},{w2},NA,NA")
            rows.append([f"{w2 - args.w1:.3f}", w2, "", ""])
        else:
            print(f"{w2 - args.w1:.3f},{w2},{Kmin:.4f},{(s_at if s_at is not None and not math.isnan(s_at) else args.thr):.3f}")
            rows.append([f"{w2 - args.w1:.3f}", w2, f"{Kmin:.6f}", f"{(s_at if s_at is not None and not math.isnan(s_at) else args.thr):.6f}"])

    # Save CSV
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_omega","omega2","Kphi_min","sync_index"])
        w.writerows(rows)

    # Save a tiny JSON summary
    summary = {
        "w1": args.w1,
        "thr": args.thr,
        "k_search": [args.kmin, args.kmax],
        "noise": args.noise,
        "eps": args.eps,
        "kv": args.kv,
        "kx": args.kx,
        "steps": args.steps,
        "burn_in": args.burn_in,
        "points": [{"delta": float(d), "omega2": float(round(args.w1+d,2)), "Kphi_min": (None if r[2]=="" else float(r[2]))} for d, r in zip(deltas, rows)]
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"# CSV:  {csv_path}", file=sys.stderr)
    print(f"# JSON: {json_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
