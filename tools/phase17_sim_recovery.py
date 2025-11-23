#!/usr/bin/env python3
"""
Phase-17: Parameter-recovery sims (no runner patches).

We fabricate Phase-8-style CSVs with known injected penalties
and verify we can recover them within tolerance.

Outputs:
  outputs/phase17/recovery_report.txt
  outputs/phase17/recovery_summary.csv
  outputs/phase17/recovery_plot.png
"""
import os, csv, math
from pathlib import Path
import matplotlib.pyplot as plt

OUTDIR = "outputs/phase17"
TOL = 0.10  # 10% relative tolerance

# Add any scenarios you like: (name, (d_tau, d_kappa, d_A))
SCENARIOS = [
    ("flat",   (5.5, 2.2, 15.8)),
    ("mild",   (4.7, 1.9, 13.5)),
    ("strong", (6.2, 2.6, 18.0)),
]

def write_phase8_csv(path, base=300.0, d_tau=5.5, d_kap=2.2, d_A=15.8):
    rows = [
        {"stem":"outputs/phase8/joint_baseline","metric":f"{base}","kind":"CHI2","source":"mock"},
        {"stem":"outputs/phase8/joint_tau0","metric":f"{base+d_tau}","kind":"CHI2","source":"mock"},
        {"stem":"outputs/phase8/joint_kappa0","metric":f"{base+d_kap}","kind":"CHI2","source":"mock"},
        {"stem":"outputs/phase8/joint_A0","metric":f"{base+d_A}","kind":"CHI2","source":"mock"},
    ]
    with open(path, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["stem","metric","kind","source"])
        w.writeheader(); w.writerows(rows)

def recover_weights(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    base = float(rows[0]["metric"])
    d = {}
    for r in rows:
        name = os.path.basename(r["stem"]).replace("joint_","").lower()
        if not r["metric"]: continue
        dy = float(r["metric"]) - base
        if name.startswith("tau"):   d["tau0"]=dy
        elif name.startswith("kappa"): d["kappa0"]=dy
        elif name.upper().startswith("A0"): d["A0"]=dy
    return d

def rel_err(rec, tru):
    return abs(rec-tru)/max(abs(tru),1e-9)

def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    summary = []
    # Generate, recover, compare
    for name,(dt,dk,dA) in SCENARIOS:
        csv_path = f"{OUTDIR}/phase8_metrics_{name}.csv"
        write_phase8_csv(csv_path, d_tau=dt, d_kap=dk, d_A=dA)
        rec = recover_weights(csv_path)
        e_tau = rel_err(rec["tau0"], dt)
        e_kap = rel_err(rec["kappa0"], dk)
        e_A   = rel_err(rec["A0"], dA)
        ok = (e_tau<=TOL and e_kap<=TOL and e_A<=TOL)
        summary.append({
            "scenario": name,
            "true_tau0": dt, "true_kappa0": dk, "true_A0": dA,
            "rec_tau0": rec["tau0"], "rec_kappa0": rec["kappa0"], "rec_A0": rec["A0"],
            "err_tau0": e_tau, "err_kappa0": e_kap, "err_A0": e_A,
            "verdict": "PASS" if ok else "FAIL"
        })

    # Write CSV + report
    with open(f"{OUTDIR}/recovery_summary.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with open(f"{OUTDIR}/recovery_report.txt","w") as f:
        f.write("Phase-17 recovery report (tolerance {:.0%})\n\n".format(TOL))
        for r in summary:
            f.write(f"{r['scenario']}: {r['verdict']}  "
                    f"(τ:{r['rec_tau0']:.3g}/{r['true_tau0']:.3g}, "
                    f"κ:{r['rec_kappa0']:.3g}/{r['true_kappa0']:.3g}, "
                    f"A:{r['rec_A0']:.3g}/{r['true_A0']:.3g})\n")
        all_pass = all(r["verdict"]=="PASS" for r in summary)
        f.write(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}\n")

    # Simple plot: recovered vs true (per scenario)
    plt.figure(figsize=(6,3.8))
    xs = range(len(summary))
    plt.plot(xs, [s["true_tau0"] for s in summary], marker="o", label="τ true")
    plt.plot(xs, [s["rec_tau0"] for s in summary],  marker="x", label="τ rec")
    plt.plot(xs, [s["true_kappa0"] for s in summary], marker="o", label="κ true")
    plt.plot(xs, [s["rec_kappa0"] for s in summary],  marker="x", label="κ rec")
    plt.plot(xs, [s["true_A0"] for s in summary], marker="o", label="A true")
    plt.plot(xs, [s["rec_A0"] for s in summary],  marker="x", label="A rec")
    plt.xticks(xs, [s["scenario"] for s in summary])
    plt.ylabel("Δ penalty")
    plt.title("Phase-17: recovery (true vs recovered)")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/recovery_plot.png", dpi=200)

    print("Wrote", f"{OUTDIR}/recovery_summary.csv")
    print("Wrote", f"{OUTDIR}/recovery_report.txt")
    print("Wrote", f"{OUTDIR}/recovery_plot.png")

if __name__ == "__main__":
    main()
