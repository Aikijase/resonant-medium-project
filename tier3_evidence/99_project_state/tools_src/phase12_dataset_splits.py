#!/usr/bin/env python3
"""
Phase-12: Dataset split evaluator.

Reads one or more Phase-8-style CSVs (columns: stem,metric,kind,source),
computes Δ vs baseline for each CSV, fits the simple penalties per lock,
and checks:
  (a) ordering A > τ > κ
  (b) magnitude drift <= 30% vs the "reference" CSV (first arg by default).

Usage examples:
  python3 tools/phase12_dataset_splits.py \
      outputs/phase8/phase8_metrics.csv \
      outputs/phase8/phase8_metrics_bao_only.csv \
      outputs/phase8/phase8_metrics_sn_only.csv \
      outputs/phase8/phase8_metrics_baosn.csv

Outputs:
  outputs/phase12/report.txt      (PASS/FAIL per split, drifts)
  outputs/phase12/summary.csv     (weights per split)
"""
import os, csv, sys, json, math
from pathlib import Path

OUTDIR = "outputs/phase12"
LOCKS = ("tau0","kappa0","A0")

def load_deltas(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows or not rows[0].get("metric"):
        return None, None
    base = float(rows[0]["metric"])
    deltas = {}
    for r in rows:
        stem = os.path.basename(r["stem"]).replace("joint_","")
        if not r["metric"]: continue
        dy = float(r["metric"]) - base
        deltas[stem] = dy
    # derive simple weights (penalty per lock) by reading the deltas directly
    # Since design is one-hot, weights ≈ observed Δ for that lock name
    weights = {
        "w_tau0":   deltas.get("tau0"),
        "w_kappa0": deltas.get("kappa0"),
        "w_A0":     deltas.get("A0"),
    }
    return deltas, weights

def check_ordering(weights):
    wA = weights.get("w_A0")
    wt = weights.get("w_tau0")
    wk = weights.get("w_kappa0")
    if any(v is None or not math.isfinite(v) for v in (wA,wt,wk)):
        return False
    return (wA > wt > wk)

def drift(a, b):
    # relative drift |a-b| / max(|b|, 1e-9)
    return (abs(a-b) / max(abs(b), 1e-9)) if (a is not None and b is not None) else float("inf")

def main(paths):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    reports = []
    summary_rows = []

    # reference = first CSV
    ref_deltas, ref_w = load_deltas(paths[0])
    if ref_deltas is None:
        print(f"[error] reference CSV has no metrics: {paths[0]}", file=sys.stderr)
        sys.exit(2)

    for i, p in enumerate(paths):
        deltas, w = load_deltas(p)
        if deltas is None:
            reports.append((p, "FAIL", "missing metrics"))
            continue
        order_ok = check_ordering(w)
        # drift checks vs reference weights
        drifts = {
            "tau0":   drift(w.get("w_tau0"),   ref_w.get("w_tau0")),
            "kappa0": drift(w.get("w_kappa0"), ref_w.get("w_kappa0")),
            "A0":     drift(w.get("w_A0"),     ref_w.get("w_A0")),
        }
        drift_ok = all(d <= 0.30 for d in drifts.values())
        verdict = "PASS" if order_ok and drift_ok else "FAIL"

        reports.append((p, verdict, f"ordering={'ok' if order_ok else 'bad'}; "
                                     f"drifts={{tau0:{drifts['tau0']:.2f}, "
                                     f"kappa0:{drifts['kappa0']:.2f}, A0:{drifts['A0']:.2f}}}"))
        summary_rows.append({
            "csv": p,
            "w_tau0": w.get("w_tau0"),
            "w_kappa0": w.get("w_kappa0"),
            "w_A0": w.get("w_A0"),
            "ordering_ok": order_ok,
            "drift_tau0": drifts["tau0"],
            "drift_kappa0": drifts["kappa0"],
            "drift_A0": drifts["A0"],
        })

    # write outputs
    with open(f"{OUTDIR}/report.txt","w") as f:
        f.write("Phase-12 dataset splits report\n\n")
        f.write(f"Reference: {paths[0]}\n\n")
        for p, verdict, note in reports:
            f.write(f"{os.path.basename(p)}: {verdict}  -- {note}\n")
    with open(f"{OUTDIR}/summary.csv","w",newline="") as f:
        wtr=csv.DictWriter(f, fieldnames=["csv","w_tau0","w_kappa0","w_A0","ordering_ok","drift_tau0","drift_kappa0","drift_A0"])
        wtr.writeheader(); wtr.writerows(summary_rows)

    print("Wrote", f"{OUTDIR}/report.txt")
    print("Wrote", f"{OUTDIR}/summary.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/phase12_dataset_splits.py <phase8_metrics.csv> [<other_split.csv> ...]")
        sys.exit(2)
    main(sys.argv[1:])
