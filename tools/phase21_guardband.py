#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

def load_preds(path):
    rows=list(csv.DictReader(open(path)))
    out=[]
    for r in rows:
        try:
            out.append({
                "omega2": float(r["omega2"]),
                "Kphi_edge": float(r["Kphi_edge"]),
                "kappa": float(r["curvature_kappa"]) if r["curvature_kappa"] else 0.0,
                "robust": float(r["robust_frac_nearby"]) if r["robust_frac_nearby"] else None,
                "warning": int(r["warning"])==1
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["omega2"])
    return out

def contiguous_warning_intervals(preds, min_len=2):
    intervals=[]
    i=0
    while i<len(preds):
        if preds[i]["warning"]:
            j=i
            while j+1<len(preds) and preds[j+1]["warning"]: j+=1
            if (j-i+1)>=min_len:
                intervals.append( (preds[i]["omega2"], preds[j]["omega2"]) )
            i=j+1
        else:
            i+=1
    return intervals

def write_policy(outdir, prefix, preds, intervals, kphi_margin=0.010):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # CSV summary (human)
    csv_path = Path(outdir, f"{prefix}_guardbands.csv")
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["omega2_min","omega2_max","note"])
        for a,b in intervals:
            w.writerow([f"{a:.6f}", f"{b:.6f}", "warning"])

    # JSON policy (machine)
    # Also include a simple suggestion: safe Kphi = edge + margin
    pol = {
        "source": "p20_predictions.csv",
        "kphi_margin": kphi_margin,
        "intervals": [{"omega2_min": a, "omega2_max": b} for a,b in intervals]
    }
    json_path = Path(outdir, f"{prefix}_guardbands.json")
    json.dump(pol, open(json_path,"w"), indent=2)

    # min report
    with open(Path(outdir, f"{prefix}_report.md"),"w") as f:
        f.write("# Phase 21 — Guardband Policy\n")
        f.write(f"- intervals: {len(intervals)}\n")
        for a,b in intervals:
            f.write(f"  - [{a:.6f}, {b:.6f}]\n")
        f.write(f"- kphi_margin: {kphi_margin}\n")
    return str(csv_path), str(json_path)

# --- Validator ---
def lerp(x0,y0,x1,y1,x):
    if x1==x0: return y0
    t=(x-x0)/(x1-x0)
    return y0+t*(y1-y0)

def interp(xs, ys, x):
    import bisect
    i=max(1, min(len(xs)-1, bisect.bisect_left(xs, x)))
    return lerp(xs[i-1], ys[i-1], xs[i], ys[i], x)

def load_curve(path):
    rows=list(csv.DictReader(open(path))); rows.sort(key=lambda r: float(r["omega2"]))
    w2=[float(r["omega2"]) for r in rows]
    K =[float(r["Kphi"]) for r in rows]
    return w2, K

def in_guardband(intervals, w2):
    for it in intervals:
        if it["omega2_min"] <= w2 <= it["omega2_max"]:
            return True
    return False

def main():
    ap=argparse.ArgumentParser(description="Phase-21 Guardband Policy & Validator")
    ap.add_argument("--preds-csv", default="outputs/phase20/p20_predictions.csv")
    ap.add_argument("--curve-csv", default="outputs/phase19/p19_edge_curve.csv")
    ap.add_argument("--outdir", default="outputs/phase21")
    ap.add_argument("--prefix", default="p21")
    ap.add_argument("--min-run", type=int, default=2, help="min consecutive warning points to form a band")
    ap.add_argument("--kphi-margin", type=float, default=0.010)
    ap.add_argument("--check-omega2", type=float, help="optional single-check mode with current Kphi")
    ap.add_argument("--check-Kphi", type=float)
    args=ap.parse_args()

    preds=load_preds(args.preds_csv)
    bands=contiguous_warning_intervals(preds, min_len=args.min_run)
    csv_path, json_path = write_policy(args.outdir, args.prefix, preds, bands, args.kphi_margin)

    # If no single-check requested, just print summary
    if args.check_omega2 is None:
        print(json.dumps({
            "guardbands_csv": csv_path,
            "guardbands_json": json_path,
            "n_intervals": len(bands)
        }, indent=2))
        return 0

    # Single check
    import json as _json
    pol = _json.load(open(json_path))
    warn = in_guardband(pol["intervals"], args.check_omega2)
    w2_axis, K_axis = load_curve(args.curve_csv)
    K_edge = interp(w2_axis, K_axis, args.check_omega2)
    suggestion = None
    if warn:
        suggestion = {"Kphi_recommend": round(K_edge + args.kphi_margin, 6),
                      "reason": "inside guardband; add margin above edge"}
    print(_json.dumps({
        "omega2": round(args.check_omega2,6),
        "Kphi_input": args.check_Kphi,
        "in_guardband": bool(warn),
        "Kphi_edge": round(K_edge,6),
        "suggestion": suggestion
    }, indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
