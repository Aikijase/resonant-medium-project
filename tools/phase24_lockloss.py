#!/usr/bin/env python3
import csv, math, json, pathlib, argparse, subprocess
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_drift(drift_csv):
    rows=list(csv.DictReader(open(drift_csv)))
    pts=[]
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"])
            km=float(r["Kphi_mid"])
            dk = r.get("dK_domega2_smooth") or r.get("dK_domega2") or ""
            dk = float(dk) if dk not in ("","nan") else math.nan
            pts.append({"target":t,"omega2":w2,"Kphi_mid":km,"dK_domega2":dk})
        except Exception:
            pass
    return pts

def grid_matrix(pts, value_key):
    targets=sorted({p["target"] for p in pts})
    w2s=sorted({p["omega2"] for p in pts})
    ti={t:i for i,t in enumerate(targets)}
    wi={w:i for i,w in enumerate(w2s)}
    M=np.full((len(targets), len(w2s)), np.nan)
    for p in pts:
        v=p[value_key]
        M[ti[p["target"]], wi[p["omega2"]]] = v
    return targets, w2s, M

def make_report(md_path, pdf_path, resource_path, params):
    md = []
    md.append("# Phase-24 — Lock-Loss Forecasting\n")
    md.append(f"- Drift input: `{params['drift_csv']}`  \n")
    md.append(f"- Assumptions: $R={params['R']}$ (max $dK_\\phi/dt$),  $v={params['v']}$ ( $d\\omega^2/dt$ )  \n")
    md.append("\n## Method\n")
    md.append("We estimate the required actuator rate per point as ")
    md.append("$R_{\\text{req}} = |dK_\\phi/d\\omega^2|\\cdot v$.  If $R_{\\text{req}} > R$, ")
    md.append("the controller cannot keep up and the lock is at risk.\n")
    # figures
    figs = [
        ("p24_required_rate_heatmap.png", "Required rate $R_{\\text{req}}$ heatmap"),
        ("p24_lockmap.png", "Feasible vs at-risk map (given R, v)")
    ]
    for fn, title in figs:
        p = pathlib.Path(resource_path)/fn
        if p.exists():
            md.append(f"\n## {title}\n\n![]({fn})\n")
    md_txt = "".join(md)
    pathlib.Path(md_path).write_text(md_txt, encoding="utf-8")
    cmd = [
        "pandoc","-f","markdown+tex_math_dollars",md_path,
        "-o", pdf_path,"--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path", str(resource_path)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Wrote:", pdf_path)

def main():
    ap=argparse.ArgumentParser(description="Phase-24: lock-loss forecast vs actuator bandwidth")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--R", type=float, default=6.0, help="actuator cap for dK_phi/dt")
    ap.add_argument("--v", type=float, default=0.01, help="environment sweep speed d(omega^2)/dt")
    ap.add_argument("--outdir", default="outputs/phase24")
    ap.add_argument("--prefix", default="p24")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    pts = read_drift(args.drift_csv)
    if not pts:
        raise SystemExit("No drift points found.")
    # build |dK/dw2| matrix
    targets, w2s, M = grid_matrix(pts, "dK_domega2")
    A = np.abs(M) * args.v  # R_req
    # lock map
    feasible = (A <= args.R)

    # write CSV summaries
    req_csv = outdir/(args.prefix+"_required_rate.csv")
    with open(req_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["target","omega2","R_req"])
        for i,t in enumerate(targets):
            for j,w2 in enumerate(w2s):
                v=A[i,j]
                if np.isfinite(v):
                    w.writerow([t,w2,f"{v:.6g}"])

    lock_csv = outdir/(args.prefix+"_lockmap.csv")
    with open(lock_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["target","omega2","feasible"])
        for i,t in enumerate(targets):
            for j,w2 in enumerate(w2s):
                v=feasible[i,j]
                if np.isfinite(A[i,j]):
                    w.writerow([t,w2,"yes" if v else "no"])

    # plots
    # 1) Required rate heatmap
    fig1=plt.figure()
    im=plt.imshow(A, aspect="auto", origin="lower",
                  extent=[min(w2s), max(w2s), min(targets), max(targets)])
    plt.colorbar(label=r"$R_{\mathrm{req}}=|dK_\phi/d\omega^2|\cdot v$")
    plt.xlabel(r"$\omega^2$"); plt.ylabel("target")
    plt.title("Required actuator rate")
    req_png = outdir/(args.prefix+"_required_rate_heatmap.png")
    fig1.savefig(req_png, dpi=160, bbox_inches="tight"); plt.close(fig1)

    # 2) Feasible vs at-risk map
    fig2=plt.figure()
    mask=np.where(np.isfinite(A), feasible.astype(float), np.nan)
    plt.imshow(mask, aspect="auto", origin="lower",
               extent=[min(w2s), max(w2s), min(targets), max(targets)])
    plt.xlabel(r"$\omega^2$"); plt.ylabel("target")
    plt.title(f"Feasible map (R={args.R}, v={args.v})  — 1=OK, 0=at-risk")
    lock_png = outdir/(args.prefix+"_lockmap.png")
    fig2.savefig(lock_png, dpi=160, bbox_inches="tight"); plt.close(fig2)

    # report
    md = outdir/(args.prefix+"_report.md")
    pdf= outdir/(args.prefix+"_report.pdf")
    make_report(str(md), str(pdf), str(outdir), {
        "drift_csv": args.drift_csv, "R": args.R, "v": args.v
    })

    # manifest
    manifest = {
        "required_rate_csv": str(req_csv),
        "lockmap_csv": str(lock_csv),
        "required_rate_heatmap": str(req_png),
        "lockmap_png": str(lock_png),
        "report_pdf": str(pdf),
        "targets": targets, "omega2_grid": w2s,
        "R": args.R, "v": args.v
    }
    (outdir/(args.prefix+"_manifest.json")).write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"status":"ok", **manifest}, indent=2))

if __name__=="__main__":
    main()
