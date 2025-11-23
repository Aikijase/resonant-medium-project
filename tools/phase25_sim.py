#!/usr/bin/env python3
import csv, math, json, pathlib, argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def read_p23_drift(csv_path):
    # Expect columns: target, omega2, Kphi_mid, dK_domega2(, dK_domega2_smooth)
    rows=list(csv.DictReader(open(csv_path)))
    by=defaultdict(list)
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"])
            km=float(r["Kphi_mid"])
            by[t].append((w2,km))
        except: pass
    # sort and dedupe
    for t in list(by.keys()):
        pts=sorted(by[t], key=lambda x:x[0])
        clean=[]
        seen=set()
        for w2,km in pts:
            if (w2) in seen: continue
            seen.add(w2); clean.append((w2,km))
        by[t]=clean
    return by

def read_p22_unc(csv_path):
    # Expect columns: target, omega2, band_width (and others)
    rows=list(csv.DictReader(open(csv_path)))
    bw=defaultdict(list)
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"])
            bwv=float(r["band_width"])
            bw[t].append((w2,bwv))
        except: pass
    for t in list(bw.keys()):
        bw[t]=sorted(bw[t], key=lambda x:x[0])
    return bw

def make_interp(xs, ys):
    xs=np.asarray(xs); ys=np.asarray(ys)
    def f(x):
        x=float(x)
        if x<=xs[0]: return float(ys[0])
        if x>=xs[-1]: return float(ys[-1])
        i=np.searchsorted(xs, x)-1
        x0,x1=xs[i],xs[i+1]
        y0,y1=ys[i],ys[i+1]
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

def simulate_one(target, f_kmid, f_bw, w2_start, w2_end, v, R, dt,
                 deadband_frac, slip_frac, K_init=None, outdir=None, prefix="p25"):
    # Forward integration until reaching w2_end or slip
    n_steps = max(1, int(abs(w2_end - w2_start)/(abs(v)*dt)))
    t = 0.0
    w2 = w2_start
    K  = f_kmid(w2) if K_init is None else K_init
    rows=[]
    slip_idx = -1

    for step in range(n_steps+1):
        km   = f_kmid(w2)
        bw   = max(1e-6, f_bw(w2))
        half = 0.5*bw
        e    = km - K
        h    = deadband_frac*half
        sthr = slip_frac*half

        # control: proportional request, rate-limited ("velocity" clamp)
        # simple bang-proportional hybrid
        if abs(e) > h:
            # proportional demand
            kp = 1.0  # unit gain; rate limit dominates
            dK_req = kp*e
            dK = max(-R, min(R, dK_req))  # rate limit [units of K per second]
        else:
            dK = 0.0

        rows.append({
            "t":t, "omega2":w2, "Kphi":K, "Kphi_mid":km,
            "band_half":half, "err":e, "act_cmd":dK, "slip":"no"
        })

        # slip test AFTER logging current state
        if abs(e) >= sthr and slip_idx<0:
            rows[-1]["slip"]="yes"
            slip_idx = step
            break

        # integrate one step
        K  = K + dK*dt
        w2 = w2 + v*dt
        t  = t + dt

        # termination if passed end
        if (w2_end - w2_start) > 0 and w2 > w2_end: break
        if (w2_end - w2_start) < 0 and w2 < w2_end: break

    # summary
    end = rows[-1]
    slipped = (end["slip"]=="yes")
    t_slip = end["t"] if slipped else math.nan
    w2_slip = end["omega2"] if slipped else math.nan
    return rows, {"target":target,"w2_start":w2_start,"w2_end":w2_end,"v":v,"R":R,
                  "dt":dt,"deadband_frac":deadband_frac,"slip_frac":slip_frac,
                  "slipped":slipped,"t_slip":t_slip,"w2_slip":w2_slip}

def write_timeseries(rows, path):
    fn=pathlib.Path(path); fn.parent.mkdir(parents=True, exist_ok=True)
    with open(fn,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def plot_timeseries(rows, png_path, title):
    import matplotlib.pyplot as plt
    t=[r["t"] for r in rows]
    w2=[r["omega2"] for r in rows]
    K=[r["Kphi"] for r in rows]
    Km=[r["Kphi_mid"] for r in rows]
    B=[r["band_half"] for r in rows]
    e=[r["err"] for r in rows]
    plt.figure()
    plt.plot(t, K, label="Kphi")
    plt.plot(t, Km, label="Kphi_mid")
    # shade band: Km ± B
    tnp=np.array(t); km=np.array(Km); bb=np.array(B)
    plt.fill_between(tnp, km-bb, km+bb, alpha=0.2, label="band")
    plt.xlabel("t"); plt.ylabel("Kphi / Kphi_mid")
    plt.title(title); plt.legend()
    plt.savefig(png_path, dpi=160, bbox_inches="tight"); plt.close()

def main():
    ap=argparse.ArgumentParser(description="Phase-25: controller policy simulation and time-to-slip")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--uncertainty-csv", default="outputs/phase22/p22_uncertainty.csv")
    ap.add_argument("--outdir", default="outputs/phase25")
    ap.add_argument("--prefix", default="p25")
    # sim params
    ap.add_argument("--target", type=float, default=0.94)
    ap.add_argument("--w2-start", type=float, default=2.74)
    ap.add_argument("--w2-end", type=float, default=2.92)
    ap.add_argument("--v", type=float, default=0.01, help="d(omega^2)/dt")
    ap.add_argument("--R", type=float, default=6.0, help="max dKphi/dt")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--deadband-frac", type=float, default=0.30, help="h = frac * (band_width/2)")
    ap.add_argument("--slip-frac", type=float, default=0.90, help="declare slip at frac * (band_width/2)")
    # batch convenience
    ap.add_argument("--batch-targets", type=str, default="", help="comma list (overrides --target)")
    ap.add_argument("--batch-R", type=str, default="", help="comma list (overrides --R)")
    ap.add_argument("--plot", action="store_true")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    p23_by = read_p23_drift(args.drift_csv)
    p22_bw = read_p22_unc(args.uncertainty_csv)

    targets = [args.target] if not args.batch_targets else [float(x) for x in args.batch_targets.split(",")]
    Rs = [args.R] if not args.batch_R else [float(x) for x in args.batch_R.split(",")]

    summary=[]
    for t in targets:
        f_kmid, f_bw, lo, hi = build_lookup(p23_by, p22_bw, t)
        w2_s = max(lo, min(hi, args.w2_start))
        w2_e = max(lo, min(hi, args.w2_end))
        for R in Rs:
            rows, meta = simulate_one(
                t, f_kmid, f_bw, w2_s, w2_e, args.v, R, args.dt,
                args.deadband_frac, args.slip_frac, K_init=None
            )
            tag=f"t{t:.2f}_R{R:g}_w{w2_s:.2f}-{w2_e:.2f}_v{args.v:g}"
            ts_csv = outdir/(args.prefix+f"_{tag}_timeseries.csv")
            write_timeseries(rows, ts_csv)
            if args.plot:
                png = outdir/(args.prefix+f"_{tag}_plot.png")
                plot_timeseries(rows, png, f"target={t}, R={R}, v={args.v}")
            meta["timeseries_csv"]=str(ts_csv)
            if args.plot: meta["plot_png"]=str(png)
            summary.append(meta)

    # write summary
    sum_csv = outdir/(args.prefix+"_summary.csv")
    if summary:
        keys=list(summary[0].keys())
        with open(sum_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)

    # manifest
    man = {
      "summary_csv": str(sum_csv),
      "outdir": str(outdir),
      "drift_csv": args.drift_csv,
      "uncertainty_csv": args.uncertainty_csv,
      "note": "timeseries files are named by target/R window; slip=yes row marks first slip event"
    }
    (outdir/(args.prefix+"_manifest.json")).write_text(json.dumps(man,indent=2))
    print(json.dumps({"status":"ok", **man}, indent=2))

if __name__=="__main__":
    main()
