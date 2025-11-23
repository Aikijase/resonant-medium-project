#!/usr/bin/env python3
import argparse, csv, bisect, json
from pathlib import Path

# ---------- Core loaders & helpers ----------
def load_curve(path):
    rows = list(csv.DictReader(open(path)))
    # sort by omega2 for safe interpolation
    rows.sort(key=lambda r: float(r["omega2"]))
    w2  = [float(r["omega2"]) for r in rows]
    K   = [float(r["Kphi"])   for r in rows]
    kap = [float(r["kappa"])  for r in rows]
    s   = [float(r["s"])      for r in rows]
    return {"rows":rows, "w2":w2, "K":K, "kap":kap, "s":s}

def load_robust(path):
    p = Path(path)
    if not p.exists(): return None
    rows = list(csv.DictReader(open(p)))
    S  = [float(r["s"]) for r in rows]
    rf = [float(r["robust_frac"]) for r in rows]
    return {"S":S, "rf":rf}

def lerp(x0,y0,x1,y1,x):
    if x1==x0: return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t*(y1-y0)

def interp(xs, ys, x):
    i = max(1, min(len(xs)-1, bisect.bisect_left(xs, x)))
    return lerp(xs[i-1], ys[i-1], xs[i], ys[i], x)

def predict(curve, robust, w2, si_thresh=0.95, kappa_warn=50.0, robust_warn=0.90):
    Khat   = interp(curve["w2"], curve["K"]  , w2)
    khat   = interp(curve["w2"], curve["kap"], w2)
    shat   = interp(curve["w2"], curve["s"]  , w2)
    rfhat  = None
    if robust:
        rfhat = interp(robust["S"], robust["rf"], shat)
    warn = (khat >= kappa_warn) or (rfhat is not None and rfhat < robust_warn)
    return {
        "omega2": round(w2,6),
        "Kphi_edge": round(Khat,6),
        "curvature_kappa": khat,
        "robust_frac_nearby": rfhat,
        "warning": bool(warn),
        "params": {"si_thresh": si_thresh, "kappa_warn": kappa_warn, "robust_warn": robust_warn}
    }

# ---------- Sweep + plots ----------
def sweep(curve, robust, w2min, w2max, w2step, si_thresh, kappa_warn, robust_warn):
    outdir = Path("outputs/phase20"); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "p20_predictions.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega2","Kphi_edge","curvature_kappa","robust_frac_nearby","warning"])
        pts=[]
        x=w2min
        while x<=w2max+1e-12:
            r = predict(curve, robust, x, si_thresh, kappa_warn, robust_warn)
            w.writerow([r["omega2"], r["Kphi_edge"], f'{r["curvature_kappa"]:.6e}',
                        "" if r["robust_frac_nearby"] is None else f'{r["robust_frac_nearby"]:.6f}',
                        int(r["warning"])])
            pts.append((r["omega2"], r["Kphi_edge"], r["robust_frac_nearby"]))
            x += w2step

    # quick plots (best-effort)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs=[p[0] for p in pts]; Ks=[p[1] for p in pts]
        plt.figure(figsize=(6,4)); plt.plot(xs,Ks,"-")
        plt.xlabel("omega2"); plt.ylabel("Kphi (edge)"); plt.title("Predicted edge Kphi vs omega2")
        plt.tight_layout(); plt.savefig(outdir / "p20_plot_kphi.png",dpi=160); plt.close()

        rs=[(p[2] if p[2] is not None else float("nan")) for p in pts]
        plt.figure(figsize=(6,3)); plt.plot(xs,rs,"-")
        plt.xlabel("omega2"); plt.ylabel("robust_frac (nearby)"); plt.title("Robustness vs omega2")
        plt.tight_layout(); plt.savefig(outdir / "p20_plot_robust.png",dpi=160); plt.close()
    except Exception:
        pass

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Phase-20 Edge Predictor + Guardband")
    ap.add_argument("--curve-csv",   default="outputs/phase19/p19_edge_curve.csv")
    ap.add_argument("--robust-csv",  default="outputs/phase19/p19_edgeR_curve_robust.csv")
    ap.add_argument("--omega2", type=float, help="single query")
    ap.add_argument("--w2-min", type=float, help="sweep: start")
    ap.add_argument("--w2-max", type=float, help="sweep: end")
    ap.add_argument("--w2-step", type=float, default=0.005, help="sweep: step")
    ap.add_argument("--si-thresh",  type=float, default=0.95)
    ap.add_argument("--kappa-warn", type=float, default=50.0)
    ap.add_argument("--robust-warn",type=float, default=0.90)
    args = ap.parse_args()

    Path("outputs/phase20").mkdir(parents=True, exist_ok=True)
    C = load_curve(args.curve_csv)
    R = load_robust(args.robust_csv)

    if args.omega2 is not None:
        out = predict(C, R, args.omega2, args.si_thresh, args.kappa_warn, args.robust_warn)
        print(json.dumps(out, indent=2))
        return 0

    if args.w2_min is not None and args.w2_max is not None:
        sweep(C, R, args.w2_min, args.w2_max, args.w2_step,
              args.si_thresh, args.kappa_warn, args.robust_warn)
        print("wrote: outputs/phase20/p20_predictions.csv and plots")
        return 0

    print("Provide either --omega2 X or --w2-min A --w2-max B [--w2-step S]", file=sys.stderr)
    return 2

if __name__=="__main__":
    raise SystemExit(main())
