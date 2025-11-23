#!/usr/bin/env python3
"""
Phase-21 → pointwise guardbands (resumable + checkpointing).

Builds canonical guardband points from a range CSV (omega2_min/max).
Writes/updates a checkpoint CSV/JSON **after every solved point** so you can
Ctrl-C and resume later with --resume.

Outputs (canonical for Phase-22):
  - <outdir>/p21_guardbands_points.csv
  - <outdir>/p21_guardbands_points.json
  (both updated incrementally)

Usage (typical):
  python3 tools/phase21_make_pointwise_guardbands.py \
    --ranges-csv outputs/phase21/p21_guardbands.csv \
    --outdir outputs/phase21 \
    --targets 0.90,0.94,0.96,0.98 \
    --omega2-step 0.02 --Kmin 0.40 --Kmax 1.60 \
    --steps 8000 --burn-in 200 --coarse-n 21 --resume

You can stop anytime; rerun with --resume to continue.
"""

import argparse, csv, json, math, pathlib, subprocess, sys, time
from typing import List, Tuple, Optional, Dict

PY = sys.executable

def run_si(omega2: float, Kphi: float, *, preset="neuron", kv=0.10, kx=0.20, eps=0.06,
           noise=0.01, adapt_every=15, steps=8000, burn_in=200,
           seed1=0, seed2=1, prefix="p21_resolve") -> float:
    args = [
        PY, "tools/phase10_phasecouple_demo.py",
        "--preset", str(preset),
        "--omega2", str(omega2),
        "--kv", str(kv), "--kx", str(kx), "--Kphi", str(Kphi),
        "--eps", str(eps), "--adapt_every", str(adapt_every),
        "--noise", str(noise),
        "--steps", str(steps), "--burn_in", str(burn_in),
        "--seed1", str(seed1), "--seed2", str(seed2),
        "--prefix", str(prefix)
    ]
    p = subprocess.run(args, capture_output=True, text=True)
    p.check_returncode()
    out = json.loads(p.stdout)
    return float(out["metrics"]["sync_index"])

def coarse_scan(omega2: float, Kmin: float, Kmax: float, n: int, target: float, cfg) -> List[Tuple[float,float]]:
    xs = [Kmin + i*(Kmax-Kmin)/(n-1) for i in range(n)]
    out=[]
    for K in xs:
        si = run_si(omega2, K, **cfg)
        out.append((K, si))
    return out

def find_brackets(samples: List[Tuple[float,float]], target: float) -> List[Tuple[float,float]]:
    br=[]
    for (K0, s0), (K1, s1) in zip(samples[:-1], samples[1:]):
        d0, d1 = s0-target, s1-target
        if d0 == 0: br.append((K0, K0))
        elif d0*d1 < 0: br.append((K0, K1))
    return br

def bisection(omega2: float, a: float, b: float, target: float, tol: float, max_iters: int, cfg) -> Optional[float]:
    fa = run_si(omega2, a, **cfg) - target
    fb = run_si(omega2, b, **cfg) - target
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0: return None
    for _ in range(max_iters):
        m = 0.5*(a+b)
        fm = run_si(omega2, m, **cfg) - target
        if abs(fm) <= tol or abs(b-a) <= max(tol, 1e-6):
            return m
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

def solve_two(omega2: float, target: float, Kmin: float, Kmax: float, cfg,
              coarse_n=21, tol=5e-4, max_iters=40) -> Tuple[Optional[float], Optional[float]]:
    samples = coarse_scan(omega2, Kmin, Kmax, coarse_n, target, cfg)
    brackets = find_brackets(samples, target)
    if not brackets:
        # widen once
        span = (Kmax-Kmin)
        Kmin2, Kmax2 = max(0.0, Kmin-0.5*span), Kmax+0.5*span
        samples = coarse_scan(omega2, Kmin2, Kmax2, coarse_n, target, cfg)
        brackets = find_brackets(samples, target)
        if not brackets:
            return (None, None)
    # unique & left→right
    uniq=[]
    for a,b in brackets:
        if not uniq or abs(a-uniq[-1][0])>1e-6 or abs(b-uniq[-1][1])>1e-6:
            uniq.append((a,b))
    uniq.sort(key=lambda ab: 0.5*(ab[0]+ab[1]))
    roots=[]
    for a,b in uniq[:2]:
        r = bisection(omega2, a, b, target, tol, max_iters, cfg)
        if r is not None:
            roots.append(r)
    if not roots:
        return (None, None)
    roots.sort()
    if len(roots)==1:
        midK = 0.5*(Kmin+Kmax)
        return (roots[0], None) if roots[0] <= midK else (None, roots[0])
    return (roots[0], roots[-1])

def load_done(checkpoint_csv: pathlib.Path) -> Dict[Tuple[float,float], Dict]:
    done={}
    if checkpoint_csv.exists():
        with checkpoint_csv.open(newline="") as f:
            r=csv.DictReader(f)
            for row in r:
                t=float(row["target"]); w2=float(row["omega2"])
                done[(t,w2)] = {"Kphi_lo": float(row["Kphi_lo"]), "Kphi_hi": float(row["Kphi_hi"])}
    return done

def write_row(checkpoint_csv: pathlib.Path, row: Dict, write_header_if_new=True):
    new_file = not checkpoint_csv.exists()
    with checkpoint_csv.open("a", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["target","omega2","Kphi_lo","Kphi_hi"])
        if write_header_if_new and new_file:
            w.writeheader()
        w.writerow(row)

def dump_json_from_csv(csv_path: pathlib.Path, json_path: pathlib.Path):
    rows=[]
    with csv_path.open(newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append({"target": float(row["target"]), "omega2": float(row["omega2"]),
                         "Kphi_lo": float(row["Kphi_lo"]), "Kphi_hi": float(row["Kphi_hi"])})
    with json_path.open("w") as f:
        json.dump(sorted(rows, key=lambda r:(r["target"], r["omega2"])), f, indent=2)

def main():
    ap=argparse.ArgumentParser(description="Phase-21: build pointwise guardbands (resumable)")
    ap.add_argument("--ranges-csv", default="outputs/phase21/p21_guardbands.csv")
    ap.add_argument("--outdir", default="outputs/phase21")
    ap.add_argument("--targets", default="0.90,0.94,0.96,0.98")
    ap.add_argument("--omega2-step", type=float, default=0.02)
    ap.add_argument("--Kmin", type=float, default=0.40)
    ap.add_argument("--Kmax", type=float, default=1.60)
    ap.add_argument("--coarse-n", type=int, default=21)
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--resume", action="store_true", help="Resume from existing points CSV if present")
    # sim knobs
    ap.add_argument("--preset", default="neuron")
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--burn-in", type=int, default=200)
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    points_csv = outdir / "p21_guardbands_points.csv"
    points_json= outdir / "p21_guardbands_points.json"

    # read ranges CSV and build omega2 grid
    with open(args.ranges_csv, newline="") as f:
        r=csv.DictReader(f)
        rows=list(r)
        if not rows: raise SystemExit("ranges CSV is empty.")
    wpairs=[]
    for row in rows:
        wmin=wmax=None
        for k,v in row.items():
            lk=k.lower()
            if "omega" in lk and "min" in lk and v not in ("",None):
                wmin=float(v)
            if "omega" in lk and "max" in lk and v not in ("",None):
                wmax=float(v)
        if wmin is not None and wmax is not None:
            if wmax<wmin: wmin,wmax=wmax,wmin
            wpairs.append((wmin,wmax))
    if not wpairs:
        raise SystemExit("No omega2_min/max found in ranges CSV.")
    w2_min=min(a for a,b in wpairs); w2_max=max(b for a,b in wpairs)
    n_steps=max(2, int(round((w2_max-w2_min)/args.omega2_step))+1)
    w2_grid=[w2_min + i*(w2_max-w2_min)/(n_steps-1) for i in range(n_steps)]
    targets=[float(x) for x in args.targets.split(",") if x.strip()]

    # resume
    done = load_done(points_csv) if args.resume else {}
    cfg=dict(preset=args.preset, kv=args.kv, kx=args.kx, eps=args.eps,
             noise=args.noise, adapt_every=15, steps=args.steps, burn_in=args.burn_in)

    total=len(targets)*len(w2_grid); solved=0
    # count already done
    for t in targets:
        for w2 in w2_grid:
            if (t,w2) in done: solved+=1

    start=time.time()
    print(f"[p21] targets={targets}  w2_range=[{w2_min:.3f},{w2_max:.3f}] "
          f"grid={len(w2_grid)}  resume={args.resume}  done={solved}/{total}")

    for t in targets:
        for w2 in w2_grid:
            key=(t,w2)
            if key in done:
                continue
            Klo,Khi = solve_two(w2, t, args.Kmin, args.Kmax, cfg,
                                coarse_n=args.coarse_n, tol=args.tol, max_iters=args.max_iters)
            if Klo is None and Khi is None:
                # skip unsolved
                continue
            if Klo is None:
                mid=0.5*(args.Kmin+args.Kmax); Klo=max(args.Kmin, min(args.Kmax, 2*mid-Khi))
            if Khi is None:
                mid=0.5*(args.Kmin+args.Kmax); Khi=max(args.Kmin, min(args.Kmax, 2*mid-Klo))
            row={"target": t, "omega2": w2, "Kphi_lo": float(Klo), "Kphi_hi": float(Khi)}
            write_row(points_csv, row)
            done[key]= {"Kphi_lo":Klo,"Kphi_hi":Khi}

            # update json snapshot each time
            dump_json_from_csv(points_csv, points_json)

            solved+=1
            if solved % 5 == 0:
                elapsed=time.time()-start
                print(f"[p21] progress {solved}/{total}  elapsed={elapsed/60:.1f}m")

    # final message
    n_rows = 0
    if points_csv.exists():
        with open(points_csv, newline="") as f:
            n_rows = sum(1 for _ in f) - 1 if f else 0
    print(json.dumps({
        "status":"ok",
        "targets": targets,
        "n_points": n_rows,
        "omega2_range": [w2_min, w2_max],
        "csv": str(points_csv),
        "json": str(points_json)
    }, indent=2))

if __name__ == "__main__":
    main()
