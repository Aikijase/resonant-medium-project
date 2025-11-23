#!/usr/bin/env python3
import csv, math, pathlib, subprocess, json
from collections import defaultdict

def read_points(csv_path):
    rows=list(csv.DictReader(open(csv_path)))
    by=defaultdict(list)
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"])
            km=float(r["Kphi_mid"])
            dk=r.get("dK_domega2_smooth") or r.get("dK_domega2") or ""
            dk=float(dk) if dk not in ("","nan") else math.nan
            by[t].append({"omega2":w2,"Kphi_mid":km,"dk":dk})
        except Exception:
            pass
    for t in by: by[t].sort(key=lambda x:x["omega2"])
    return by

def write_feasibility(by, B, out_csv):
    fieldnames=["target","omega2","Kphi_mid","dK_domega2","abs_drift","feasible"]
    with open(out_csv,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for t,lst in sorted(by.items()):
            for p in lst:
                v=p["dk"]; feas=(math.isfinite(v) and abs(v)<=B)
                w.writerow({
                    "target":t,
                    "omega2":p["omega2"],
                    "Kphi_mid":p["Kphi_mid"],
                    "dK_domega2":(f"{v:.6g}" if math.isfinite(v) else ""),
                    "abs_drift":(f"{abs(v):.6g}" if math.isfinite(v) else ""),
                    "feasible":"yes" if feas else ("no" if math.isfinite(v) else "")
                })

def plot_overlay(by, B, out_png):
    import matplotlib.pyplot as plt
    plt.figure()
    for t,lst in sorted(by.items()):
        x=[p["omega2"] for p in lst]
        y=[p["dk"] for p in lst]
        plt.plot(x,y,marker="o",label=f"target={t:g}")
        xin=[xx for xx,yy in zip(x,y) if (math.isfinite(yy) and abs(yy)>B)]
        yin=[yy for xx,yy in zip(x,y) if (math.isfinite(yy) and abs(yy)>B)]
        if xin: plt.scatter(xin,yin,marker="x")
    plt.axhline(B,linestyle="--"); plt.axhline(-B,linestyle="--")
    plt.xlabel(r"$\omega^2$"); plt.ylabel(r"$dK_\phi/d\omega^2$")
    plt.title(f"Drift with feasibility bands (|drift| ≤ {B})")
    plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()

def main():
    import argparse
    ap=argparse.ArgumentParser(description="Phase-23 feasibility vs bandwidth cap B")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--B", type=float, default=10.0, help="bandwidth cap for |dK/dω²|")
    ap.add_argument("--outdir", default="outputs/phase23")
    ap.add_argument("--prefix", default="p23")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    by=read_points(args.drift_csv)

    feas_csv = outdir/(args.prefix+"_feasibility.csv")
    feas_png = outdir/(args.prefix+"_drift_feasibility.png")
    write_feasibility(by, args.B, str(feas_csv))
    plot_overlay(by, args.B, str(feas_png))

    # re-render main report (it will embed the PNG if present)
    subprocess.run([
        "python3","tools/phase23_make_report.py",
        "--uncertainty-csv","outputs/phase22/p22_uncertainty.csv",
        "--drift-csv", str(args.drift_csv)
    ], check=True)

    print(json.dumps({
        "status":"ok","B":args.B,
        "feasibility_csv": str(feas_csv),
        "overlay_png": str(feas_png),
        "report_pdf": "outputs/phase23/p23_report.pdf"
    }, indent=2))

if __name__=="__main__":
    main()
