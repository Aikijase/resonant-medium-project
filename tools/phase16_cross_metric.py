#!/usr/bin/env python3
"""
Phase-16: Cross-metric agreement.

Reads a Phase-8-style CSV (stem,metric,kind,source) for the reference set
(typically outputs/phase8/phase8_metrics.csv). By default uses CHI2 deltas.
Optionally derives ΔAIC and ΔBIC from Δχ² with provided Δk per-lock and n.

Outputs:
  outputs/phase16/report.txt
  outputs/phase16/summary.csv
Prints RESULT: PASS/FAIL (PASS if ordering A>τ>κ holds for all selected metrics).
"""
import os, csv, math, argparse
from pathlib import Path

OUTDIR = "outputs/phase16"

def read_deltas(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    base = float(rows[0]["metric"])
    d = {}
    for r in rows:
        stem = os.path.basename(r["stem"]).replace("joint_","").lower()
        if not r["metric"]: continue
        dy = float(r["metric"]) - base
        if stem.startswith("tau"):   d["tau0"]   = dy
        elif stem.startswith("kappa"): d["kappa0"] = dy
        elif stem.upper().startswith("A0"): d["A0"] = dy
    return d  # {'tau0': Δχ²_tau, 'kappa0': ..., 'A0': ...}

def ordering_ok(d):
    return (d.get("A0") is not None and d.get("tau0") is not None and d.get("kappa0") is not None
            and d["A0"] > d["tau0"] > d["kappa0"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Phase-8 metrics CSV (reference)")
    ap.add_argument("--check", default="chi2,aic,bic",
                    help="comma list of metrics to check: chi2,aic,bic (default all)")
    ap.add_argument("--dk", default="tau0=-1,kappa0=-1,A0=-1",
                    help="delta-k per lock when deriving AIC/BIC, e.g. 'tau0=-1,kappa0=-1,A0=-1'")
    ap.add_argument("--n", type=int, default=1000, help="effective sample size for BIC (ln n term)")
    args = ap.parse_args()

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    checks = [c.strip().lower() for c in args.check.split(",") if c.strip()]
    dkmap = {"tau0":0,"kappa0":0,"A0":0}
    for part in args.dk.split(","):
        if "=" in part:
            k,v = part.split("=",1)
            k=k.strip(); v=v.strip()
            if k in dkmap:
                try: dkmap[k]=float(v)
                except: pass

    d_chi = read_deltas(args.csv)
    results = []
    all_pass = True

    # χ²
    if "chi2" in checks:
        ok = ordering_ok(d_chi)
        all_pass &= ok
        results.append(("chi2", d_chi, ok))

    # AIC: ΔAIC = Δχ² + 2Δk
    if "aic" in checks:
        d_aic = {k: d_chi.get(k) + 2.0*dkmap[k] for k in ("tau0","kappa0","A0")}
        ok = ordering_ok(d_aic)
        all_pass &= ok
        results.append(("aic", d_aic, ok))

    # BIC: ΔBIC = Δχ² + (ln n)Δk
    if "bic" in checks:
        lnN = math.log(max(args.n,1))
        d_bic = {k: d_chi.get(k) + lnN*dkmap[k] for k in ("tau0","kappa0","A0")}
        ok = ordering_ok(d_bic)
        all_pass &= ok
        results.append(("bic", d_bic, ok))

    # write summary
    with open(f"{OUTDIR}/summary.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","tau0","kappa0","A0","ordering_ok"])
        for name, dmap, ok in results:
            w.writerow([name, dmap["tau0"], dmap["kappa0"], dmap["A0"], ok])

    with open(f"{OUTDIR}/report.txt","w") as f:
        f.write("Phase-16 cross-metric report\n\n")
        f.write(f"Source CSV: {args.csv}\n")
        f.write(f"Δk: {dkmap}   n: {args.n}\n\n")
        for name, dmap, ok in results:
            f.write(f"{name.upper()}: ordering={'ok' if ok else 'bad'}  "
                    f"(tau0={dmap['tau0']:.6g}, kappa0={dmap['kappa0']:.6g}, A0={dmap['A0']:.6g})\n")
        f.write(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}\n")

    print("Wrote", f"{OUTDIR}/summary.csv")
    print("Wrote", f"{OUTDIR}/report.txt")
    print("RESULT:", "PASS" if all_pass else "FAIL")

if __name__ == "__main__":
    main()
