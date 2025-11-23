#!/usr/bin/env python3
import csv, math, json, pathlib, argparse, random, statistics
from collections import defaultdict
import numpy as np

# ---------- IO helpers (P23/P22) ----------
def read_p23_drift(csv_path):
    by=defaultdict(list)
    with open(csv_path) as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                t=float(row["target"]); w2=float(row["omega2"]); km=float(row["Kphi_mid"])
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
    by=defaultdict(list)
    with open(csv_path) as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                t=float(row["target"]); w2=float(row["omega2"]); bw=float(row["band_width"])
                by[t].append((w2,bw))
            except: pass
    for t in list(by):
        by[t]=sorted(by[t], key=lambda x:x[0])
    return by

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
    w2k, km = zip(*p23_by[target])
    w2b, bw = zip(*p22_bw[target])
    f_kmid = make_interp(np.array(w2k), np.array(km))
    f_bw   = make_interp(np.array(w2b), np.array(bw))
    w2_min = max(min(w2k), min(w2b))
    w2_max = min(max(w2k), max(w2b))
    return f_kmid, f_bw, w2_min, w2_max

# ---------- colored noise ----------
def OU(n, dt, theta, sigma, rng):
    x=np.zeros(n)
    for i in range(1,n):
        dx=theta*(-x[i-1])*dt + sigma*np.sqrt(dt)*rng.randn()
        x[i]=x[i-1]+dx
    return x

# ---------- endurance with soft integrator ----------
def endurance_run(f_kmid, f_bw, w2_lo, w2_hi, v_base, R, dt, fh, slip_frac,
                  seconds, beta, eta, leak, seed=0, chaos="chaotic", nonlinear=True):
    rng=np.random.RandomState(seed)
    n_steps=max(1,int(seconds/dt))

    if chaos=="light":
        ou_w = OU(n_steps+1, dt, theta=1.1, sigma=0.006, rng=rng)
        ou_k = OU(n_steps+1, dt, theta=0.9, sigma=0.004, rng=rng)
        shrink_p=0.02
    elif chaos=="medium":
        ou_w = OU(n_steps+1, dt, theta=0.9, sigma=0.012, rng=rng)
        ou_k = OU(n_steps+1, dt, theta=0.8, sigma=0.007, rng=rng)
        shrink_p=0.04
    else:
        ou_w = OU(n_steps+1, dt, theta=0.8, sigma=0.02, rng=rng)
        ou_k = OU(n_steps+1, dt, theta=0.7, sigma=0.01, rng=rng)
        shrink_p=0.06

    # disturbances schedule (like P28)
    disturb=set()
    for base in np.arange(1.0, seconds-1.0, 5.0):
        if rng.rand()<0.75: disturb.add(int(base/dt))

    # init
    w2 = 0.5*(w2_lo+w2_hi)
    K  = f_kmid(w2)
    I  = 0.0
    t=0.0

    # metrics
    t_first_slip=np.nan
    n_slips=0
    effort_L1=0.0
    chatter=0
    prev_sign=0
    abs_err_sum=0.0
    recov=[]
    recov_timer=None

    for i in range(n_steps):
        km = f_kmid(w2)
        if nonlinear:
            x=(w2 - w2_lo)/max(1e-9, w2_hi - w2_lo)
            taper = 1 - 0.25*(3*x**2 - 2*x**3)
            km = taper*km + (1-taper)*km

        half  = 0.5*max(1e-6, f_bw(w2))
        if rng.rand() < shrink_p*dt:
            half *= rng.uniform(0.7,0.9)

        h    = fh*half
        sthr = slip_frac*half

        K_meas = K + ou_k[i]
        e = km - K_meas

        # P term (rate-limited outside deadband)
        if abs(e) > h:
            dKp = max(-R, min(R, e))
        else:
            dKp = 0.0

        # I term (only integrate inside deadband; leak)
        if abs(e) <= h:
            I = leak*I + eta*e
        else:
            I = leak*I  # decay when outside band

        dK = dKp + beta*I

        # metrics
        effort_L1 += abs(dK)*dt
        sgn = 1 if dK>1e-9 else (-1 if dK<-1e-9 else 0)
        if sgn!=0 and prev_sign!=0 and sgn!=prev_sign: chatter+=1
        if sgn!=0: prev_sign=sgn
        abs_err_sum += abs(e)*dt

        # slip detect + anti-windup reset
        if abs(e) >= sthr:
            n_slips += 1
            if math.isnan(t_first_slip): t_first_slip = t
            I = 0.0  # reset integral on slip

        # recovery timer wrt deadband
        if abs(e) <= h:
            if recov_timer is not None:
                recov_timer[1] += dt
                if recov_timer[1] >= 0.5:
                    recov.append(t - recov_timer[0])
                    recov_timer=None
        else:
            if recov_timer is None:
                recov_timer=[t,0.0]

        # disturbance
        v = v_base + ou_w[i]
        if i in disturb:
            typ = rng.choice(["pause","reverse","step","shrink"])
            if typ=="reverse": v += -2*v_base
            elif typ=="step":  w2 += rng.uniform(-0.02, 0.02)
            elif typ=="shrink": half *= rng.uniform(0.5, 0.8)

        # evolve
        K += dK*dt
        w2 += v*dt
        t  += dt
        w2 = min(max(w2, w2_lo), w2_hi)

    no_slip=(n_slips==0)
    mean_abs_err = abs_err_sum/(n_steps*dt)
    rec_med = (statistics.median(recov) if recov else "")

    return {"no_slip":"yes" if no_slip else "no",
            "t_first_slip":("" if math.isnan(t_first_slip) else t_first_slip),
            "n_slips":n_slips,
            "effort_L1":effort_L1,
            "chatter":chatter,
            "mean_abs_err":mean_abs_err,
            "recovery_median_s":rec_med}

def main():
    ap=argparse.ArgumentParser(description="Phase-29: integrator endurance (bias rejection)")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--uncertainty-csv", default="outputs/phase22/p22_uncertainty.csv")
    ap.add_argument("--p27-manifest", default="outputs/phase27/p27_manifest.json")
    ap.add_argument("--outdir", default="outputs/phase29")
    ap.add_argument("--prefix", default="p29")
    ap.add_argument("--targets", default="0.94,0.98")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--v", type=float, default=0.01)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--deadband-frac", type=float, default=None)
    ap.add_argument("--slip-frac", type=float, default=0.90)
    ap.add_argument("--beta", type=float, default=0.5, help="integrator weight")
    ap.add_argument("--eta", type=float, default=0.01, help="integrator gain per second")
    ap.add_argument("--leak", type=float, default=0.998, help="integrator leak per step")
    ap.add_argument("--chaos", choices=["light","medium","chaotic"], default="chaotic")
    ap.add_argument("--nonlinear", action="store_true", default=True)
    ap.add_argument("--n-worlds", type=int, default=50)
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if pathlib.Path(args.p27_manifest).exists():
        m=json.loads(pathlib.Path(args.p27_manifest).read_text())
        rec=m.get("recommendation",{})
        R  = args.R if args.R is not None else float(rec.get("R",6.0))
        fh = args.deadband_frac if args.deadband_frac is not None else float(rec.get("deadband_frac",0.30))
        w2_s = float(rec.get("w2_start", 2.74))
        w2_e = float(rec.get("w2_end", 2.92))
    else:
        R,fh = (args.R if args.R is not None else 6.0, args.deadband_frac if args.deadband_frac is not None else 0.30)
        w2_s,w2_e = 2.74,2.92

    p23=read_p23_drift(args.drift_csv)
    p22=read_p22_unc(args.uncertainty_csv)
    targets=[float(x) for x in args.targets.split(",") if x]

    results=[]
    rnd=random.Random(2025)
    for t in targets:
        fkm, fbw, lo, hi = build_lookup(p23, p22, t)
        w2_lo=max(lo, min(hi, w2_s))
        w2_hi=max(lo, min(hi, w2_e))
        for _ in range(args.n_worlds):
            seed=rnd.randrange(1<<31)
            met=endurance_run(fkm, fbw, w2_lo, w2_hi, args.v, R, args.dt, fh, args.slip_frac,
                              args.seconds, args.beta, args.eta, args.leak,
                              seed=seed, chaos=args.chaos, nonlinear=args.nonlinear)
            results.append({"target":t,"R":R,"deadband_frac":fh,"v":args.v,"dt":args.dt,
                            "beta":args.beta,"eta":args.eta,"leak":args.leak,
                            "seconds":args.seconds,"omega2_lo":w2_lo,"omega2_hi":w2_hi,
                            "world_seed":seed, **met})

    out_csv=outdir/(args.prefix+"_endurance.csv")
    with open(out_csv,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # aggregate
    agg=[]
    by=defaultdict(list)
    for r in results: by[r["target"]].append(r)
    def med(col, rows):
        xs=[r[col] for r in rows if r[col]!=""]
        return (statistics.median(xs) if xs else "")
    for t, rows in sorted(by.items()):
        n=len(rows); ok=sum(1 for r in rows if r["no_slip"]=="yes")
        agg.append({"target":t,"%no_slip":100.0*ok/n, "runs":n,
                    "effort_L1_med":med("effort_L1",rows),
                    "chatter_med":med("chatter",rows),
                    "mean_abs_err_med":med("mean_abs_err",rows),
                    "t_first_slip_med":med("t_first_slip",rows),
                    "recovery_median_s_med":med("recovery_median_s",rows)})
    agg_csv=None
    if agg:
        agg_csv=outdir/(args.prefix+"_aggregate.csv")
        with open(agg_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            w.writeheader(); w.writerows(agg)

    manifest={"endurance_csv":str(out_csv),
              "aggregate_csv":(str(agg_csv) if agg_csv else None),
              "params":vars(args),
              "policy":{"R":R,"deadband_frac":fh,"slip_frac":args.slip_frac,
                        "w2_start":w2_s,"w2_end":w2_e}}
    (outdir/(args.prefix+"_manifest.json")).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status":"ok", **manifest}, indent=2))

if __name__=="__main__":
    main()
