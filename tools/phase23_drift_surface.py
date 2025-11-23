#!/usr/bin/env python3
import csv, json, pathlib
from collections import defaultdict

def read_unc_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            try:
                target = float(row["target"])
                w2     = float(row["omega2"])
                klo    = float(row["Kphi_lo"])
                khi    = float(row["Kphi_hi"])
                kmid   = float(row.get("Kphi_mid", (klo+khi)/2.0))
            except Exception:
                continue
            rows.append({"target":target,"omega2":w2,"Kphi_lo":klo,"Kphi_hi":khi,"Kphi_mid":kmid})
        return rows

def finite_diff(xs, ys):
    n=len(xs)
    if n==0: return []
    if n==1: return [float("nan")]
    out=[None]*n
    for i in range(n):
        if 0<i<n-1:
            dx = xs[i+1]-xs[i-1]
            out[i]=(ys[i+1]-ys[i-1])/dx if dx!=0 else float("nan")
        elif i==0:
            dx = xs[1]-xs[0]
            out[i]=(ys[1]-ys[0])/dx if dx!=0 else float("nan")
        else:
            dx = xs[-1]-xs[-2]
            out[i]=(ys[-1]-ys[-2])/dx if dx!=0 else float("nan")
    return out

def main():
    import argparse, json
    ap=argparse.ArgumentParser(description="Phase-23: compute drift surface from p22_uncertainty.csv")
    ap.add_argument("--uncertainty-csv", default="outputs/phase22/p22_uncertainty.csv")
    ap.add_argument("--outdir", default="outputs/phase23")
    ap.add_argument("--prefix", default="p23")
    args=ap.parse_args()

    rows=read_unc_csv(args.uncertainty_csv)
    if not rows:
        raise SystemExit(f"No usable rows in {args.uncertainty_csv}")

    by_t=defaultdict(list)
    for r in rows:
        by_t[r["target"]].append(r)
    for t in by_t:
        by_t[t].sort(key=lambda r:r["omega2"])

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    pts_csv=outdir/(args.prefix+"_drift_points.csv")
    surf_json=outdir/(args.prefix+"_surface.json")

    all_points=[]
    for t, lst in sorted(by_t.items()):
        w2=[r["omega2"] for r in lst]
        km=[r["Kphi_mid"] for r in lst]
        # central / one-sided finite diff
        dk=[float("nan")]*len(w2)
        if len(w2)>=2:
            for i in range(len(w2)):
                if 0<i<len(w2)-1:
                    dx=w2[i+1]-w2[i-1]
                    dk[i]=(km[i+1]-km[i-1])/dx if dx!=0 else float("nan")
                elif i==0:
                    dx=w2[1]-w2[0]
                    dk[i]=(km[1]-km[0])/dx if dx!=0 else float("nan")
                else:
                    dx=w2[-1]-w2[-2]
                    dk[i]=(km[-1]-km[-2])/dx if dx!=0 else float("nan")
        for r, d in zip(lst, dk):
            all_points.append({
                "target": t,
                "omega2": r["omega2"],
                "Kphi_mid": r["Kphi_mid"],
                "dK_domega2": d
            })

    with pts_csv.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["target","omega2","Kphi_mid","dK_domega2"])
        w.writeheader()
        for p in all_points:
            w.writerow(p)

    surf = {"points": all_points,
            "targets": sorted(list(by_t.keys())),
            "omega2_values": sorted({p["omega2"] for p in all_points})}
    surf_json.write_text(json.dumps(surf, indent=2))

    print(json.dumps({"status":"ok", "csv":str(pts_csv), "json":str(surf_json),
                      "n_points": len(all_points)}, indent=2))

if __name__=="__main__":
    main()
