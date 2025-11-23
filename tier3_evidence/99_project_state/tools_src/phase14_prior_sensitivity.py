#!/usr/bin/env python3
"""
Phase-14: Prior sensitivity check.

Input: one reference CSV (first arg) + 1..N variant CSVs, all Phase-8-style with
columns: stem, metric, kind, source.

For each CSV we:
  - compute deltas vs baseline row (first row in that CSV)
  - infer simple weights: w_tau0, w_kappa0, w_A0
  - check ordering: A > τ > κ

For each VARIANT (everything after the first/ref CSV) we also compute
relative drift vs the reference weights and require all three drifts <= 30%.

Outputs:
  outputs/phase14/report.txt
  outputs/phase14/summary.csv
Console prints a RESULT line (PASS/FAIL overall).

Usage:
  python3 tools/phase14_prior_sensitivity.py \
    outputs/phase8/phase8_metrics.csv \
    outputs/phase8/phase8_metrics_H0p5.csv \
    outputs/phase8/phase8_metrics_H0m5.csv
"""
import os, sys, csv, math
from pathlib import Path
THRESH = 0.30
OUTDIR = "outputs/phase14"

def parse_weights_from_csv(csv_path):
    """Return dict with w_tau0, w_kappa0, w_A0 (may be None if missing)."""
    import re
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except FileNotFoundError:
        return None
    if not rows or not rows[0].get("metric"):
        return None
    base = float(rows[0]["metric"])
    deltas = {}
    for r in rows:
        stem = os.path.basename(r["stem"])
        # capture tau0|kappa0|A0 anywhere (ignore prefixes/suffixes)
        m = re.search(r"(tau0|kappa0|A0)", stem, re.I)
        if not m or not r["metric"]:
            continue
        token = m.group(1).lower()
        if token == "tau0":
            key = "tau0"
        elif token == "kappa0":
            key = "kappa0"
        elif token == "a0":
            key = "A0"
        else:
            continue
        dy = float(r["metric"]) - base
        deltas[key] = dy
    return {
        "w_tau0":   deltas.get("tau0"),
        "w_kappa0": deltas.get("kappa0"),
        "w_A0":     deltas.get("A0"),
    }

def ordering_ok(w):
    try:
        return (w["w_A0"] is not None and w["w_tau0"] is not None and w["w_kappa0"] is not None
                and w["w_A0"] > w["w_tau0"] > w["w_kappa0"])
    except Exception:
        return False

def rdiff(a, b):
    # robust relative difference: if missing, count as infinite drift
    if a is None or b is None:
        return float("inf")
    return abs(a - b) / max(abs(b), 1e-9)

def main(paths):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    ref_csv = paths[0]
    ref_w = parse_weights_from_csv(ref_csv)
    if not ref_w:
        print(f"[error] bad/missing reference CSV: {ref_csv}", file=sys.stderr)
        sys.exit(2)

    summary_hdr = ["csv","is_reference","w_tau0","w_kappa0","w_A0",
                   "ordering_ok","drift_tau0","drift_kappa0","drift_A0","verdict"]
    summary_rows = []

    report_lines = [f"Phase-14 prior sensitivity report\n",
                    f"Reference: {ref_csv}\n"]

    all_pass = True

    for i, csv_path in enumerate(paths):
        w = parse_weights_from_csv(csv_path)
        if not w:
            report_lines.append(f"{os.path.basename(csv_path)}: FAIL (missing/invalid CSV)")
            summary_rows.append([csv_path, i==0, None, None, None, False, None, None, None, "FAIL"])
            all_pass = False
            continue

        order = ordering_ok(w)
        if i == 0:
            # reference: drifts = 0 by definition
            d_tau = d_kap = d_A = 0.0
            verdict = "PASS" if order else "FAIL"
            if not order:
                all_pass = False
            report_lines.append(f"{os.path.basename(csv_path)}: {verdict} (ordering={'ok' if order else 'bad'})")
        else:
            d_tau = rdiff(w["w_tau0"],   ref_w["w_tau0"])
            d_kap = rdiff(w["w_kappa0"], ref_w["w_kappa0"])
            d_A   = rdiff(w["w_A0"],     ref_w["w_A0"])
            drift_ok = (d_tau <= THRESH and d_kap <= THRESH and d_A <= THRESH)
            verdict = "PASS" if (order and drift_ok) else "FAIL"
            if verdict == "FAIL": all_pass = False
            report_lines.append(
                f"{os.path.basename(csv_path)}: {verdict} "
                f"(ordering={'ok' if order else 'bad'}; "
                f"drift tau={d_tau:.2f}, kappa={d_kap:.2f}, A={d_A:.2f})"
            )

        summary_rows.append([csv_path, i==0, w["w_tau0"], w["w_kappa0"], w["w_A0"],
                             order, d_tau, d_kap, d_A, verdict])

    # write files
    with open(f"{OUTDIR}/report.txt","w") as f:
        for ln in report_lines:
            f.write(ln + ("\n" if not ln.endswith("\n") else ""))
        f.write(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}\n")
    with open(f"{OUTDIR}/summary.csv","w",newline="") as f:
        wr = csv.writer(f); wr.writerow(summary_hdr); wr.writerows(summary_rows)

    print("Wrote", f"{OUTDIR}/report.txt")
    print("Wrote", f"{OUTDIR}/summary.csv")
    print("RESULT:", "PASS" if all_pass else "FAIL")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/phase14_prior_sensitivity.py <ref.csv> <var1.csv> [var2.csv ...]")
        sys.exit(2)
    main(sys.argv[1:])

