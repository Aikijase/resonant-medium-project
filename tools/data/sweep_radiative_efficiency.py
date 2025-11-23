#!/usr/bin/env python3
# tools/data/sweep_radiative_efficiency.py
# Sweep epsilon to match Δrho_from_accretion / observed_delta_rhoBH ~ 1.0
# Usage:
#   python3 tools/data/sweep_radiative_efficiency.py \
#     --agn-lf data/agn_lf_bolometric.prep.csv \
#     --rho-bh data/rho_bh_z.clean.csv \
#     --eps-range 0.010 0.025 --n 31 \
#     --out logs/epsilon_sweep.csv --plot plots/epsilon_sweep.png
import argparse, csv, json, math, os, subprocess, sys
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

p = argparse.ArgumentParser()
p.add_argument("--agn-lf", required=True, help="AGN LF (bolometric) CSV")
p.add_argument("--rho-bh", required=True, help="Clean BH density history CSV")
p.add_argument("--eps-range", nargs=2, type=float, required=True, metavar=("EPS_MIN","EPS_MAX"))
p.add_argument("--n", type=int, default=31)
p.add_argument("--out", required=True, help="Output CSV for sweep results")
p.add_argument("--plot", help="Optional path to PNG plot of ratio vs epsilon")
args = p.parse_args()

eps_min, eps_max = args.eps_range
N = max(2, args.n)
epsilons = [eps_min + (eps_max-eps_min)*i/(N-1) for i in range(N)]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
rows = []
for i, eps in enumerate(epsilons, 1):
    tmp_rdot = f"/tmp/rdot_eps_{i:03d}.csv"
    tmp_qc   = f"/tmp/qc_eps_{i:03d}.csv"
    # 1) Build rho_dot for this epsilon
    cmd1 = [
        sys.executable, "tools/data/agn_lf_to_accretion.py",
        args.agn_lf, "--epsilon", f"{eps:.6f}",
        "--out", tmp_rdot
    ]
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    if r1.returncode != 0:
        print("ERROR agn_lf_to_accretion failed at eps", eps, r1.stderr, file=sys.stderr)
        continue
    # 2) Run QC to extract the ratio
    cmd2 = [
        sys.executable, "tools/data/qc_consistency_rhoBH_vs_accretion.py",
        "--rho-bh", args.rho_bh,
        "--rho-dot", tmp_rdot,
        "--out", tmp_qc
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode != 0:
        print("ERROR qc_consistency failed at eps", eps, r2.stderr, file=sys.stderr)
        continue
    # Read qc csv
    qc = {}
    with open(tmp_qc) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("metric"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2:
                qc[parts[0]] = parts[1]
    def F(key, default=float("nan")):
        try: return float(qc.get(key, default))
        except: return float("nan")
    rows.append({
        "epsilon": eps,
        "delta_rho_from_accretion_Msun_Mpc3": F("delta_rho_from_accretion_Msun_Mpc3"),
        "observed_delta_rhoBH_Msun_Mpc3":     F("observed_delta_rhoBH_Msun_Mpc3"),
        "ratio_accretion_to_observed":        F("ratio_accretion_to_observed"),
    })

# Write CSV
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epsilon","ratio_accretion_to_observed","delta_rho_from_accretion_Msun_Mpc3","observed_delta_rhoBH_Msun_Mpc3"])
    for r in rows:
        w.writerow([f"{r['epsilon']:.6f}",
                    f"{r['ratio_accretion_to_observed']:.6f}",
                    f"{r['delta_rho_from_accretion_Msun_Mpc3']:.6e}",
                    f"{r['observed_delta_rhoBH_Msun_Mpc3']:.6e}"])

print(f"Wrote {args.out} rows={len(rows)}")

# Optional plot
if args.plot:
    if plt is None:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
    else:
        xs = [r["epsilon"] for r in rows if r["ratio_accretion_to_observed"]==r["ratio_accretion_to_observed"]]
        ys = [r["ratio_accretion_to_observed"] for r in rows if r["ratio_accretion_to_observed"]==r["ratio_accretion_to_observed"]]
        if xs and ys:
            plt.figure()
            plt.axhline(1.0, linestyle="--")
            plt.plot(xs, ys, marker="o")
            plt.xlabel("epsilon")
            plt.ylabel("ratio: Δρ_acc / Δρ_BH")
            os.makedirs(os.path.dirname(args.plot), exist_ok=True)
            plt.savefig(args.plot, dpi=160, bbox_inches="tight")
            print(f"Wrote {args.plot}")
