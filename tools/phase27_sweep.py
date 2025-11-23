#!/usr/bin/env python3
import csv, math, json, pathlib, argparse, random
from collections import defaultdict
import numpy as np

# --- Shared helpers (from P25/P26, trimmed) ---
def read_p23_drift(csv_path):
    rows=list(csv.DictReader(open(csv_path)))
    by=defaultdict(list)
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"]); km=float(r["Kphi_mid"])
            by[t].append((w2,km))
        except: pass
    for t in list(by):
        pts=sorted(by[t], key=lambda x:x[0])
        uniq=[]; seen=set()
        for w2,km in pts:
            if w2 in seen: continue
            seen.add(w2); uniq.append((w2,km))
        by[t]=uniq
    return by

def read_p22_unc(csv_path):
    rows=list(csv.DictReader(open(csv_path)))
    bw=defaultdict(list)
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"]); bwv=float(r["band_width"])
            bw[t].append((w2,bwv))
        except: pass
    for t in list(bw): bw[t]=sorted(bw[t], key=lambda x:x[0])
    return bw

def make_interp(xs, ys):
    xs=np.asarray(xs); ys=np.asarray(ys)
    def f(x):
        x=float(x)
        if x<=xs[0]: return float(ys[0])
        if x>=xs[-1]: return float(ys[-1])
        i=np.searchsorted(xs, x)-1
        x0,x1=xs[i],xs[i+1]; y0,y1=ys[i],ys[i+1]
        w=(x-x0)/(x1-x0)
        return float((1-w)*y0 + w*y1)
    return f

def build_lookup(p23_by, p22_bw, target):
    if target not in p23_by or target not in p22_bw:
        raise ValueError("Missing target in inputs")
    w2_k, km = zip(*p23_by[target])
    w2_b, bw = zip(*p22_bw[target])
    f_kmid = make_interp(np.array(w2_k), np.array(km))
    f_bw   = make_interp(np.array(w2_b), np.array(bw))
    w2_min = max(min(w2_k), min(w2_b))
    w2_max = min(max(w2_k), max(w2_b))
    return f_kmid, f_bw, w2_min, w2_max

def ornstein_uhlenbeck(n, dt, theta, sigma):
    x=np.zeros(n)
    for i in range(1,n):
        dx=theta*(-x[i-1])*dt + sigma*np.sqrt(dt)*np.random.randn()
        x[i]=x[i-1]+dx
    return x

def simulate_world(f_kmid, f_bw, w2_s, w2_e, v_base, R, dt, deadband_frac, slip_frac,
                   chaos="chaotic", nonlinear=True, seed=0):
    rng=np.random.RandomState(seed)
    T = abs(w2_e - w2_s)/max(1e-9, abs(v_base))
    n_steps = max(1, int(T/dt))

    if chaos=="light":
        ou_w = ornstein_uhlenbeck(n_steps+1, dt, theta=1.2, sigma=0.005)
        ou_k = ornstein_uhlenbeck(n_steps+1, dt, theta=1.0, sigma=0.003)
        shrink_prob = 0.02
    elif chaos=="medium":
        ou_w = ornstein_uhlenbeck(n_steps+1, dt, theta=1.0, sigma=0.01)
        ou_k = ornstein_uhlenbeck(n_steps+1, dt, theta=0.8, sigma=0.006)
        shrink_prob = 0.04
    else:
        ou_w = ornstein_uhlenbeck(n_steps+1, dt, theta=0.8, sigma=0.02)
        ou_k = ornstein_uhlenbeck(n_steps+1, dt, theta=0.7, sigma=0.01)
        shrink_prob = 0.06

    K = f_kmid(w2_s)
    w2 = w2_s
    t  = 0.0
    slipped=False

    for i in range(n_steps+1):
        km = f_kmid(w2)
        if nonlinear:
            x = (w2 - min(w2_s,w2_e)) / max(1e-9, abs(w2_e-w2_s))
            taper = 1 - 0.25*(3*x**2 - 2*x**3)
            km = taper*km + (1-taper)*km

        bw = max(1e-6, f_bw(w2))
        half = 0.5*bw
        if rng.rand() < shrink_prob*dt:
            half *= rng.uniform(0.7, 0.9)

        K_meas = K + ou_k[i]
        e = km - K_meas
        h = deadband_frac*half
        sthr = 0.90*half  # fix slip_frac at 0.90 for the sweep (stable compare)

        if abs(e) > h:
            dK_req = e
            dK = max(-R, min(R, dK_req))
        else:
            dK = 0.0

        if abs(e) >= sthr:
            slipped=True; break

        v = v_base + ou_w[i]
        t  += dt
        w2 += v*dt
        K  += dK*dt

        if (w2_e - w2_s) > 0 and w2 > w2_e: break
        if (w2_e - w2_s) < 0 and w2 < w2_e: break

    return not slipped  # True if no-slip

def main():
    ap=argparse.ArgumentParser(description="Phase-27: policy sweep over R and deadband_frac")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--uncertainty-csv", default="outputs/phase22/p22_uncertainty.csv")
    ap.add_argument("--outdir", default="outputs/phase27")
    ap.add_argument("--prefix", default="p27")
    ap.add_argument("--targets", default="0.94,0.98")
    ap.add_argument("--w2-start", type=float, default=2.74)
    ap.add_argument("--w2-end", type=float, default=2.92)
    ap.add_argument("--v", type=float, default=0.01)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--R-grid", default="4,5,6,7,8")
    ap.add_argument("--deadband-grid", default="0.20,0.25,0.30,0.35,0.40")
    ap.add_argument("--worlds-per-policy", type=int, default=12)
    ap.add_argument("--chaos", choices=["light","medium","chaotic"], default="chaotic")
    ap.add_argument("--nonlinear", action="store_true", default=True)
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    p23 = read_p23_drift(args.drift_csv)
    p22 = read_p22_unc(args.uncertainty_csv)
    targets=[float(x) for x in args.targets.split(",") if x]
    Rs=[float(x) for x in args.R_grid.split(",") if x]
    Hs=[float(x) for x in args.deadband_grid.split(",") if x]

    # per-target lookups
    looks={}
    for t in targets:
        fkm, fbw, lo, hi = build_lookup(p23, p22, t)
        w2_s = max(lo, min(hi, args.w2_start))
        w2_e = max(lo, min(hi, args.w2_end))
        looks[t]=(fkm, fbw, w2_s, w2_e)

    # sweep
    rng=random.Random(123)
    out_rows=[]
    for t in targets:
        fkm, fbw, w2_s, w2_e = looks[t]
        for R in Rs:
            for h in Hs:
                ok=0
                for k in range(args.worlds_per_policy):
                    seed = rng.randrange(1<<31)
                    res = simulate_world(fkm, fbw, w2_s, w2_e, args.v, R, args.dt,
                                         h, 0.90, chaos=args.chaos, nonlinear=args.nonlinear, seed=seed)
                    ok += 1 if res else 0
                pct = 100.0*ok/args.worlds_per_policy
                out_rows.append({
                    "target":t,"R":R,"deadband_frac":h,"v":args.v,"dt":args.dt,
                    "w2_start":w2_s,"w2_end":w2_e,"worlds":args.worlds_per_policy,
                    "pct_no_slip":pct,"chaos":args.chaos
                })

    # write CSV
    sweep_csv = outdir/(args.prefix+"_policy_map.csv")
    with open(sweep_csv,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # pick a Pareto-ish recommendation: max pct_no_slip, then min R, then max deadband (calmer)
    best = sorted(out_rows, key=lambda r:(-r["pct_no_slip"], r["R"], -r["deadband_frac"]))[0]

    man={
      "policy_map_csv": str(sweep_csv),
      "recommendation": best,
      "params": vars(args)
    }
    (outdir/(args.prefix+"_manifest.json")).write_text(json.dumps(man, indent=2))
    print(json.dumps({"status":"ok", **man}, indent=2))

if __name__=="__main__":
    main()
