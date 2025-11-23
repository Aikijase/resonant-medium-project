#!/usr/bin/env python3
"""
Phase-13: Redshift-bin evaluation

Reads 2+ Phase-8-style CSVs (columns: stem, metric, kind, source), each one
representing a redshift bin. For each CSV:
- compute Δ vs baseline
- infer weights (penalty per lock): w_tau0, w_kappa0, w_A0
- check ordering: A > τ > κ

Across bins:
- check each weight is either "flat" (<=30% rel spread) or "monotone" (all diffs share a sign)
- if any weight is neither flat nor monotone -> mark 'noisy'

PASS criteria:
- every bin passes ordering
- no weight is 'noisy' across bins

Usage:
  python3 tools/phase13_redshift_bins.py \
      outputs/phase8/phase8_metrics_z1.csv \
      outputs/phase8/phase8_metrics_z2.csv \
      outputs/phase8/phase8_metrics_z3.csv
"""
import os, sys, csv, math, json
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = "outputs/phase13"
LOCKS  = ("tau0","kappa0","A0")

def load_weights(csv_path):
    import re, os, csv
    rows = list(csv.DictReader(open(csv_path)))
    if not rows or not rows[0].get("metric"):
        return None
    base = float(rows[0]["metric"])
    deltas = {}
    for r in rows:
        stem = os.path.basename(r["stem"])
        # strip 'joint_' prefix and pull the lock token regardless of suffixes like _z2/_z3
        m = re.search(r'(tau0|kappa0|A0)', stem, re.I)
        if not m:
            continue
        token = m.group(1)
        # normalise to keys we expect
        if token.lower() == "a0":
            key = "A0"
        elif token.lower() == "tau0":
            key = "tau0"
        else:
            key = "kappa0"
        if not r["metric"]:
            continue
        dy = float(r["metric"]) - base
        deltas[key] = dy

    w = {
        "w_tau0":   deltas.get("tau0"),
        "w_kappa0": deltas.get("kappa0"),
        "w_A0":     deltas.get("A0"),
    }
    return w

def ordering_ok(w):
    try:
        return w["w_A0"] > w["w_tau0"] > w["w_kappa0"]
    except Exception:
        return False

def trend_kind(vals, tol=0.30):
    """Return 'flat' if rel spread <= tol, 'monotone' if all diffs same sign,
       else 'noisy'."""
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if len(vals) <= 1: return "flat"
    mean_abs = max(sum(abs(v) for v in vals)/len(vals), 1e-9)
    rel_spread = (max(vals) - min(vals)) / mean_abs
    if rel_spread <= tol: return "flat"
    diffs = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
    pos = all(d>=0 for d in diffs)
    neg = all(d<=0 for d in diffs)
    if pos or neg: return "monotone"
    return "noisy"

def main(paths):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    # derive labels from filenames: ..._zXXX.csv -> zXXX, else basename
    labels = []
    W = []  # list of weight dicts per bin
    row_out = []

    for p in paths:
        w = load_weights(p)
        label = os.path.splitext(os.path.basename(p))[0].replace("phase8_metrics_","")
        labels.append(label)
        ok = ordering_ok(w) if w else False
        W.append(w)
        row_out.append({
            "csv": p,
            "label": label,
            "w_tau0": None if not w else w["w_tau0"],
            "w_kappa0": None if not w else w["w_kappa0"],
            "w_A0": None if not w else w["w_A0"],
            "ordering_ok": ok
        })

    # trend assessment across bins
    seq_tau  = [r["w_tau0"]   for r in row_out]
    seq_kap  = [r["w_kappa0"] for r in row_out]
    seq_A    = [r["w_A0"]     for r in row_out]
    trend_tau = trend_kind(seq_tau)
    trend_kap = trend_kind(seq_kap)
    trend_A   = trend_kind(seq_A)

    pass_bins = all(r["ordering_ok"] for r in row_out)
    pass_trend = all(t in ("flat","monotone") for t in (trend_tau, trend_kap, trend_A))
    verdict = "PASS" if (pass_bins and pass_trend) else "FAIL"

    # write summary CSV
    with open(f"{OUTDIR}/summary.csv","w",newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["label","w_tau0","w_kappa0","w_A0","ordering_ok"])
        wtr.writeheader()
        for r in row_out: wtr.writerow({k:r[k] for k in wtr.fieldnames})

    # write report
    with open(f"{OUTDIR}/report.txt","w") as f:
        f.write("Phase-13 redshift-bins report\n\n")
        for r in row_out:
            f.write(f"{r['label']}: ordering={'ok' if r['ordering_ok'] else 'bad'}; "
                    f"weights: tau0={r['w_tau0']}, kappa0={r['w_kappa0']}, A0={r['w_A0']}\n")
        f.write("\nTrend assessment across bins:\n")
        f.write(f"  tau0:   {trend_tau}\n")
        f.write(f"  kappa0: {trend_kap}\n")
        f.write(f"  A0:     {trend_A}\n")
        f.write(f"\nOVERALL: {verdict}\n")

    # plot trends (non-GUI save)
    try:
        import matplotlib.pyplot as plt
        x = list(range(len(labels)))
        plt.figure(figsize=(7,4))
        plt.plot(x, seq_tau,  marker="o", label="τ penalty")
        plt.plot(x, seq_kap,  marker="o", label="κ penalty")
        plt.plot(x, seq_A,    marker="o", label="A penalty")
        plt.xticks(x, labels, rotation=15)
        plt.ylabel("Δ penalty")
        plt.title("Phase-13: penalties vs redshift bin")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/weights_trend.png", dpi=200)
    except Exception:
        pass

    print("Wrote", f"{OUTDIR}/summary.csv")
    print("Wrote", f"{OUTDIR}/report.txt")
    print("Wrote", f"{OUTDIR}/weights_trend.png")
    print("RESULT:", verdict)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 tools/phase13_redshift_bins.py <metrics_z1.csv> <metrics_z2.csv> [metrics_z3.csv ...]")
        sys.exit(2)
    main(sys.argv[1:])
