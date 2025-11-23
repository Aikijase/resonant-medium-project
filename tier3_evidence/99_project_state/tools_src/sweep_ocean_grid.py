#!/usr/bin/env python3
"""
Phase-7 Ocean: grid sweep over (mu0, nu0)
Runs:
  - tools/phase7_ocean/ode_growth_ocean.py  -> fs8_pred_ocean
  - tools/phase6_resonant_response_v5.py   -> R(z) and plot
Then summarizes: inside-band count, fraction, median, 10–90% for R.

Outputs:
  <out_dir>/summary.csv
  <out_dir>/summary.md
  Per-combo CSV/JSON/PNG for Phase-6 results
"""
import argparse, os, sys, subprocess, itertools, datetime
from pathlib import Path
import numpy as np
import pandas as pd

def parse_list(s):
    # Accept "0.98,1.00,1.02" or "0.98:1.02:0.01" (start:stop:step inclusive)
    s = s.strip()
    if ":" in s:
        a,b,c = map(float, s.split(":"))
        n = int(round((b - a)/c)) + 1
        vals = [a + i*c for i in range(n)]
        # avoid FP glitches
        return [float(f"{v:.10g}") for v in vals]
    return [float(x) for x in s.split(",") if x.strip()]

def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("ERROR running:", " ".join(cmd), file=sys.stderr)
        print(r.stdout, file=sys.stderr); print(r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv", default="outputs/phase2/fs8_eval.csv")
    ap.add_argument("--out-dir", default=None, help="Default: outputs/phase7/sweeps/<timestamp>")
    ap.add_argument("--mu0", default="0.98,1.00,1.02")
    ap.add_argument("--nu0", default="0.00,0.03,0.05,0.08")
    ap.add_argument("--sigma8", type=float, default=0.81)
    ap.add_argument("--Om", type=float, default=0.3)

    ap.add_argument("--z-min", type=float, default=0.42)
    ap.add_argument("--z-max", type=float, default=0.75)
    ap.add_argument("--max-gap", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.003)
    ap.add_argument("--clip-min", type=float, default=0.6)
    ap.add_argument("--clip-max", type=float, default=1.4)

    args = ap.parse_args()
    mu_list = parse_list(args.mu0)
    nu_list = parse_list(args.nu0)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"outputs/phase7/sweeps/sweep_{ts}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for mu0, nu0 in itertools.product(mu_list, nu_list):
        tag = f"mu{mu0:.3f}_nu{nu0:.3f}".replace(".","p")
        o_fs8 = Path(out_dir) / f"fs8_eval_ocean_{tag}.csv"
        o_csv = Path(out_dir) / f"resonant_response_{tag}.csv"
        o_json= Path(out_dir) / f"resonant_response_{tag}.json"
        o_png = Path(out_dir) / f"resonant_response_{tag}.png"

        # 1) Build model fs8 via ocean ODE
        cmd1 = [
            "python3","tools/phase7_ocean/ode_growth_ocean.py",
            "--ref-csv", args.ref_csv,
            "--out-csv", str(o_fs8),
            "--mu0", str(mu0),
            "--nu0", str(nu0),
            "--sigma8", str(args.sigma8),
            "--Om", str(args.Om),
        ]
        run(cmd1)

        # 2) Phase-6 reporter on that model
        cmd2 = [
            "python3","tools/phase6_resonant_response_v5.py",
            "--fs8-csv", str(o_fs8),
            "--z-col","z","--lcdm-col","fs8_fit","--model-col","fs8_pred_ocean",
            "--z-min", str(args.z_min), "--z-max", str(args.z_max),
            "--max-gap", str(args.max_gap), "--eps", str(args.eps),
            "--floor-frac","0.0",
            "--clip-min", str(args.clip_min), "--clip-max", str(args.clip_max),
            "--out-csv", str(o_csv), "--out-json", str(o_json), "--out-png", str(o_png)
        ]
        run(cmd2)

        # 3) Metrics
        d = pd.read_csv(o_csv)
        R = d["R_res"].to_numpy()
        m = np.isfinite(R)
        total = int(m.sum())
        inside = int(((R[m]>=args.clip_min)&(R[m]<=args.clip_max)).sum())
        low = int((R[m]<args.clip_min).sum())
        high= int((R[m]>args.clip_max).sum())
        med = float(np.nanmedian(R))
        q10,q90 = [float(x) for x in np.nanquantile(R,[.1,.9])]
        frac = inside/total if total>0 else 0.0

        rows.append({
            "mu0": mu0, "nu0": nu0,
            "inside": inside, "total": total, "frac_in": frac,
            "low": low, "high": high,
            "median_R": med, "q10_R": q10, "q90_R": q90,
            "csv": str(o_csv), "png": str(o_png)
        })
        print(f"[{tag}] inside={inside}/{total}  median={med:.3f}  10–90%=[{q10:.3f},{q90:.3f}]")

    # Summary table
    summ = pd.DataFrame(rows)
    # rank: maximize frac_in, then minimize |median-1|, then minimize spread
    summ["rank_key"] = list(zip(
        -summ["frac_in"].values, np.abs(summ["median_R"].values-1.0), (summ["q90_R"]-summ["q10_R"]).values
    ))
    summ = summ.sort_values("rank_key").drop(columns=["rank_key"])
    csv_path = Path(out_dir)/"summary.csv"
    md_path  = Path(out_dir)/"summary.md"
    summ.to_csv(csv_path, index=False)

    # Markdown summary
    with open(md_path,"w") as f:
        f.write(f"# Ocean sweep summary ({ts})\n\n")
        f.write(f"Window: z∈[{args.z_min},{args.z_max}]  eps={args.eps}  max-gap={args.max_gap}\n\n")
        f.write(summ.to_markdown(index=False))
        f.write("\n")
    # Print top few
    print("\nTop results:")
    print(summ.head(8).to_string(index=False))
    print(f"\nWrote {csv_path}\nWrote {md_path}")

if __name__ == "__main__":
    main()
