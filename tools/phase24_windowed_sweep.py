#!/usr/bin/env python3
import csv, math, json, pathlib, argparse, subprocess
import numpy as np
import matplotlib.pyplot as plt

def read_req(drift_csv, w2_lo, w2_hi, targets_sel):
    rows=list(csv.DictReader(open(drift_csv)))
    pts=[]
    for r in rows:
        try:
            t=float(r["target"]); w2=float(r["omega2"])
            if not (w2_lo <= w2 <= w2_hi): continue
            if targets_sel and (t not in targets_sel): continue
            dk=r.get("dK_domega2_smooth") or r.get("dK_domega2") or ""
            dk=float(dk) if dk not in ("","nan") else math.nan
            if math.isfinite(dk): pts.append((t,w2,abs(dk)))
        except: pass
    targets=sorted({t for t,_,_ in pts})
    w2s=sorted({w for _,w,_ in pts})
    ti={t:i for i,t in enumerate(targets)}
    wi={w:i for i,w in enumerate(w2s)}
    M=np.full((len(targets),len(w2s)), np.nan)
    for t,w2,val in pts: M[ti[t], wi[w2]] = val
    return targets, w2s, M

def main():
    ap=argparse.ArgumentParser(description="Phase-24: windowed R×v sweep (% feasible)")
    ap.add_argument("--drift-csv", default="outputs/phase23/p23_drift_points.csv")
    ap.add_argument("--w2-lo", type=float, required=True)
    ap.add_argument("--w2-hi", type=float, required=True)
    ap.add_argument("--targets", type=str, default="", help="comma list, e.g. 0.90,0.94")
    ap.add_argument("--R-min", type=float, default=4.0)
    ap.add_argument("--R-max", type=float, default=12.0)
    ap.add_argument("--R-steps", type=int, default=17)
    ap.add_argument("--v-min", type=float, default=0.005)
    ap.add_argument("--v-max", type=float, default=0.03)
    ap.add_argument("--v-steps", type=int, default=11)
    ap.add_argument("--outdir", default="outputs/phase24")
    ap.add_argument("--prefix", default="p24w")
    args=ap.parse_args()

    targets_sel=[float(x) for x in args.targets.split(",") if x] if args.targets else []
    outdir=pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    targets, w2s, A = read_req(args.drift_csv, args.w2_lo, args.w2_hi, targets_sel)
    mask=np.isfinite(A); n_tot=int(mask.sum())
    if n_tot==0: raise SystemExit("No points in requested window/targets.")

    Rs=np.linspace(args.R_min, args.R_max, args.R_steps)
    vs=np.linspace(args.v_min, args.v_max, args.v_steps)
    P=np.zeros((len(vs), len(Rs)))

    for i,v in enumerate(vs):
        Req = A * v
        ok_base = mask & np.isfinite(Req)
        for j,R in enumerate(Rs):
            ok = ok_base & (Req <= R)
            P[i,j] = 100.0 * ok.sum() / n_tot

    # CSV
    sweep_csv = outdir/(args.prefix+"_feasible_pct.csv")
    with open(sweep_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["v\\R"] + [f"{R:.3g}" for R in Rs])
        for i,v in enumerate(vs):
            w.writerow([f"{v:.4g}"] + [f"{P[i,j]:.2f}" for j in range(len(Rs))])

    # Heatmap
    fig=plt.figure()
    im=plt.imshow(P, aspect="auto", origin="lower",
                  extent=[Rs.min(), Rs.max(), vs.min(), vs.max()], vmin=0, vmax=100)
    plt.colorbar(label="% feasible (windowed)")
    plt.xlabel(r"Actuator cap $R$ (max $dK_\phi/dt$)")
    plt.ylabel(r"Sweep speed $v$ ($d\omega^2/dt$)")
    title = f"% feasible on ω²∈[{args.w2_lo:.2f},{args.w2_hi:.2f}]"
    if targets_sel:
        title += f", targets={targets_sel}"
    plt.title(title)
    heat_png = outdir/(args.prefix+"_feasible_pct_heatmap.png")
    fig.savefig(heat_png, dpi=160, bbox_inches="tight"); plt.close(fig)

    # Report
    md = outdir/(args.prefix+"_report.md")
    pdf= outdir/(args.prefix+"_report.pdf")
    content = (
        "# Phase-24 — Windowed R×v sweep\n"
        f"- Drift: `{args.drift_csv}`\n"
        f"- Window: $\\omega^2\\in[{args.w2_lo},{args.w2_hi}]$\n"
        + (f"- Targets: {targets_sel}\n" if targets_sel else "")
        + f"- R∈[{args.R_min},{args.R_max}] ({args.R_steps} steps), "
          f"v∈[{args.v_min},{args.v_max}] ({args.v_steps} steps)\n\n"
        "## Feasible percentage (windowed)\n\n"
        f"![]({heat_png.name})\n"
    )
    md.write_text(content, encoding="utf-8")

    subprocess.run([
        "pandoc","-f","markdown+tex_math_dollars",str(md),
        "-o", str(pdf),"--pdf-engine","xelatex","-V","geometry:margin=20mm",
        "--resource-path", str(outdir)
    ], check=True)

    # Manifest
    manifest = {
        "heatmap_png": str(heat_png),
        "sweep_csv": str(sweep_csv),
        "report_pdf": str(pdf),
        "w2_lo": args.w2_lo, "w2_hi": args.w2_hi,
        "targets": targets_sel
    }
    (outdir/(args.prefix+"_manifest.json")).write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"status":"ok", **manifest}, indent=2))

if __name__ == "__main__":
    main()
