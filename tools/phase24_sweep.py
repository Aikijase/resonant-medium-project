#!/usr/bin/env python3
import csv, math, json, pathlib, argparse, subprocess
import numpy as np
import matplotlib.pyplot as plt

def read_req(drift_csv):
    # returns unique (targets, w2s) and |dK/dw2| matrix
    rows=list(csv.DictReader(open(drift_csv)))
    pts=[]
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"])
            dk=r.get("dK_domega2_smooth") or r.get("dK_domega2") or ""
            dk=float(dk) if dk not in ("","nan") else math.nan
            pts.append((t,w2,abs(dk)))
        except: pass
    targets=sorted({t for t,_,_ in pts})
    w2s=sorted({w for _,w,_ in pts})
    ti={t:i for i,t in enumerate(targets)}
    wi={w:i for i,w in enumerate(w2s)}
    M=np.full((len(targets),len(w2s)), np.nan)
    for t,w2,val in pts:
        M[ti[t], wi[w2]] = val
    return targets, w2s, M

def main():
    ap=argparse.ArgumentParser(description="Phase-24: sweep R×v → % feasible")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--R-min", type=float, default=4.0)
    ap.add_argument("--R-max", type=float, default=12.0)
    ap.add_argument("--R-steps", type=int, default=17)   # 4..12 step 0.5
    ap.add_argument("--v-min", type=float, default=0.005)
    ap.add_argument("--v-max", type=float, default=0.03)
    ap.add_argument("--v-steps", type=int, default=11)   # 0.005..0.03
    ap.add_argument("--outdir", default="outputs/phase24")
    ap.add_argument("--prefix", default="p24s")
    args=ap.parse_args()

    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    targets, w2s, A = read_req(args.drift_csv)  # A is |dK/dw2|
    mask=np.isfinite(A); n_tot=mask.sum()

    Rs=np.linspace(args.R_min, args.R_max, args.R_steps)
    vs=np.linspace(args.v_min, args.v_max, args.v_steps)
    P=np.zeros((len(vs), len(Rs)))  # % feasible for each (v,R)

    for i,v in enumerate(vs):
        Req = A * v
        for j,R in enumerate(Rs):
            ok = (Req <= R) & mask
            P[i,j] = 100.0 * ok.sum() / n_tot if n_tot>0 else np.nan

    # write CSV
    sweep_csv = outdir/(args.prefix+"_feasible_pct.csv")
    with open(sweep_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["v\\R"] + [f"{R:.3g}" for R in Rs])
        for i,v in enumerate(vs):
            w.writerow([f"{v:.4g}"] + [f"{P[i,j]:.2f}" for j in range(len(Rs))])

    # heatmap
    import matplotlib.pyplot as plt
    fig=plt.figure()
    im=plt.imshow(P, aspect="auto", origin="lower",
                  extent=[Rs.min(), Rs.max(), vs.min(), vs.max()], vmin=0, vmax=100)
    plt.colorbar(label="% feasible")
    plt.xlabel(r"Actuator cap $R$ (max $dK_\phi/dt$)")
    plt.ylabel(r"Sweep speed $v$ ($d\omega^2/dt$)")
    plt.title("% feasible over ($\omega^2$, target)")
    heat_png = outdir/(args.prefix+"_feasible_pct_heatmap.png")
    fig.savefig(heat_png, dpi=160, bbox_inches="tight"); plt.close(fig)

    # tiny report
    md = outdir/(args.prefix+"_report.md")
    pdf= outdir/(args.prefix+"_report.pdf")
    md.write_text(
        "# Phase-24 — R×v sweep\n"
        f"- Drift input: `{args.drift_csv}`\n"
        f"- R∈[{args.R_min},{args.R_max}] ({args.R_steps} steps), "
        f"v∈[{args.v_min},{args.v_max}] ({args.v_steps} steps)\n\n"
        "## Feasible percentage heatmap\n\n"
        f"![]({heat_png.name})\n",
        encoding="utf-8"
    )
    subprocess.run([
        "pandoc","-f","markdown+tex_math_dollars",str(md),
        "-o", str(pdf),"--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path", str(outdir)
    ], check=True)

    (outdir/(args.prefix+"_manifest.json")).write_text(json.dumps({
        "heatmap_png": str(heat_png),
        "sweep_csv": str(sweep_csv),
        "report_pdf": str(pdf),
        "R_grid": Rs.tolist(),
        "v_grid": vs.tolist()
    }, indent=2))

    print(json.dumps({"status":"ok",
        "heatmap_png": str(heat_png),
        "sweep_csv": str(sweep_csv),
        "report_pdf": str(pdf)}, indent=2))

if __name__=="__main__":
    main()
