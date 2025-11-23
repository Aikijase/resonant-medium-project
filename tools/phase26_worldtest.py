#!/usr/bin/env python3
import csv, math, json, pathlib, argparse, random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ---------- Input helpers ----------
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

# ---------- World generator ----------
def ornstein_uhlenbeck(n, dt, theta, sigma, x0=0.0):
    x=np.zeros(n); x[0]=x0
    for i in range(1,n):
        dx=theta*(-x[i-1])*dt + sigma*np.sqrt(dt)*np.random.randn()
        x[i]=x[i-1]+dx
    return x

def simulate_world(f_kmid, f_bw, w2_s, w2_e, v_base, R, dt,
                   deadband_frac, slip_frac,
                   seed, chaos_level="chaotic",
                   nonlinear=True, save_series=False, series_cap=10000):
    rng=np.random.RandomState(seed)
    # duration from base sweep
    T = abs(w2_e - w2_s)/max(1e-9, abs(v_base))
    n_steps = max(1, int(T/dt))
    # perturb profiles
    if chaos_level=="light":
        ou_w = ornstein_uhlenbeck(n_steps+1, dt, theta=1.2, sigma=0.005)
        ou_k = ornstein_uhlenbeck(n_steps+1, dt, theta=1.0, sigma=0.003)
        shrink_prob = 0.02
    elif chaos_level=="medium":
        ou_w = ornstein_uhlenbeck(n_steps+1, dt, theta=1.0, sigma=0.01)
        ou_k = ornstein_uhlenbeck(n_steps+1, dt, theta=0.8, sigma=0.006)
        shrink_prob = 0.04
    else:  # chaotic
        ou_w = ornstein_uhlenbeck(n_steps+1, dt, theta=0.8, sigma=0.02)
        ou_k = ornstein_uhlenbeck(n_steps+1, dt, theta=0.7, sigma=0.01)
        shrink_prob = 0.06

    # random bursts on w2 and K measurement
    jump_w2_times = set(rng.choice(np.arange(50,n_steps,50), size=max(1,n_steps//400), replace=False))
    jump_k_times  = set(rng.choice(np.arange(80,n_steps,80), size=max(1,n_steps//500), replace=False))

    t=0.0
    w2=w2_s
    km=f_kmid(w2)
    K = km  # start centered
    rows=[]
    slip=False; slip_t=math.nan; slip_w2=math.nan

    for i in range(n_steps+1):
        # non-linear drift in Kmid (optional cubic taper on ends)
        km = f_kmid(w2)
        if nonlinear:
            # soften ends: blend with a smoothed version near edges
            # use normalized x in [0,1] across [w2_s,w2_e]
            x = (w2 - min(w2_s,w2_e)) / max(1e-9, abs(w2_e-w2_s))
            taper = 1 - 0.25*(3*x**2 - 2*x**3)  # smooth curve ~1 in middle, <1 at ends
            km = taper*km + (1-taper)*km  # keeps km but enforces edge gentleness

        bw = max(1e-6, f_bw(w2))
        half = 0.5*bw

        # effective band shrink (occasional)
        if rng.rand() < shrink_prob*dt:
            half *= rng.uniform(0.7, 0.9)  # temporary narrowing

        # measured K with colored noise + occasional jump
        K_meas = K + ou_k[i]
        if i in jump_k_times: K_meas += rng.uniform(-0.05,0.05)

        e = km - K_meas
        h = deadband_frac*half
        sthr = slip_frac*half

        # rate-limited proportional action on the true K (actuator acts on true, senses noisy)
        if abs(e) > h:
            kp = 1.0
            dK_req = kp*e
            dK = max(-R, min(R, dK_req))
        else:
            dK = 0.0

        # log state before slip decision
        if save_series and len(rows) < series_cap:
            rows.append({
                "t":t,"omega2":w2,"Kphi":K,"Kphi_mid":km,"band_half":half,
                "K_meas":K_meas,"err_meas":e,"act_cmd":dK,"slip":"no"
            })

        # slip decision on measured error
        if abs(e) >= sthr and not slip:
            slip=True; slip_t=t; slip_w2=w2
            if save_series and len(rows)>0:
                rows[-1]["slip"]="yes"
            break

        # environment update: v with OU + bursts
        v = v_base + ou_w[i]
        if i in jump_w2_times: v += rng.uniform(-0.02, 0.02)
        w2 = w2 + v*dt
        t  = t + dt
        K  = K + dK*dt

        # terminate when overshoot end
        if (w2_e - w2_s) > 0 and w2 > w2_e: break
        if (w2_e - w2_s) < 0 and w2 < w2_e: break

    meta = {
        "slipped": "yes" if slip else "no",
        "t_slip": slip_t if slip else "",
        "w2_slip": slip_w2 if slip else "",
        "n_steps": i+1
    }
    return rows, meta

# ---------- Batch driver ----------
def main():
    ap=argparse.ArgumentParser(description="Phase-26: world stress testing")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--uncertainty-csv", default="outputs/phase22/p22_uncertainty.csv")
    ap.add_argument("--outdir", default="outputs/phase26")
    ap.add_argument("--prefix", default="p26")
    ap.add_argument("--targets", default="0.94,0.98")
    ap.add_argument("--w2-start", type=float, default=2.74)
    ap.add_argument("--w2-end", type=float, default=2.92)
    ap.add_argument("--v", type=float, default=0.01)
    ap.add_argument("--R", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--deadband-frac", type=float, default=0.30)
    ap.add_argument("--slip-frac", type=float, default=0.90)
    ap.add_argument("--n-worlds", type=int, default=50)
    ap.add_argument("--chaos", choices=["light","medium","chaotic"], default="chaotic")
    ap.add_argument("--nonlinear", action="store_true", default=True)
    ap.add_argument("--save-series", action="store_true")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    p23_by = read_p23_drift(args.drift_csv)
    p22_bw = read_p22_unc(args.uncertainty_csv)

    targets = [float(x) for x in args.targets.split(",") if x]
    # build lookups
    lookups={}
    for t in targets:
        f_kmid, f_bw, lo, hi = build_lookup(p23_by, p22_bw, t)
        # clamp start/end to overlap range
        w2_s = max(lo, min(hi, args.w2_start))
        w2_e = max(lo, min(hi, args.w2_end))
        lookups[t]=(f_kmid, f_bw, w2_s, w2_e)

    # run worlds
    summary=[]
    rng=random.Random(42)
    for t in targets:
        f_kmid,f_bw,w2_s,w2_e = lookups[t]
        for w in range(args.n_worlds):
            seed = rng.randrange(1<<31)
            rows, meta = simulate_world(
                f_kmid, f_bw, w2_s, w2_e, args.v, args.R, args.dt,
                args.deadband_frac, args.slip_frac,
                seed=seed, chaos_level=args.chaos,
                nonlinear=args.nonlinear, save_series=args.save_series
            )
            rec={
                "target":t,"R":args.R,"v":args.v,"dt":args.dt,
                "deadband_frac":args.deadband_frac,"slip_frac":args.slip_frac,
                "w2_start":w2_s,"w2_end":w2_e,
                "world_seed":seed, **meta
            }
            summary.append(rec)
            # save a few timeseries for flavor
            if args.save_series and rows:
                tag=f"t{t:.2f}_w{w:03d}"
                ts_csv = outdir/(args.prefix+f"_{tag}_timeseries.csv")
                with open(ts_csv,"w",newline="") as f:
                    wcsv=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    wcsv.writeheader(); wcsv.writerows(rows)

    # write summary
    sum_csv = outdir/(args.prefix+"_worlds.csv")
    if summary:
        keys=list(summary[0].keys())
        with open(sum_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)

    # quick stats png
    try:
        import pandas as pd
        df=pd.read_csv(sum_csv)
        by=df.groupby("target")["slipped"].apply(lambda s:(s=="no").mean()*100).reset_index(name="pct_no_slip")
        plt.figure()
        plt.bar(by["target"], by["pct_no_slip"])
        plt.xlabel("target"); plt.ylabel("% no-slip")
        plt.title("Phase-26: no-slip rate across worlds")
        bar_png = outdir/(args.prefix+"_noslip_bar.png")
        plt.savefig(bar_png, dpi=160, bbox_inches="tight"); plt.close()
    except Exception:
        bar_png = None

    manifest={
        "worlds_csv": str(sum_csv),
        "noslip_bar_png": (str(bar_png) if bar_png else None),
        "params": vars(args)
    }
    (outdir/(args.prefix+"_manifest.json")).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status":"ok", **manifest}, indent=2))

if __name__=="__main__":
    main()
