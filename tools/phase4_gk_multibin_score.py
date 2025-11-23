#!/usr/bin/env python3
"""
Phase-4: g×κ multi-bin scorer (overall + per-survey aggregates).

Input CSV (default: data/isw/gk_bins.csv):
  survey,bin_id,z_eff,A_lit,sigma_lit[,notes]

Model amplitude (shape-only prediction) provided via:
  --scale <float> OR --scale-file outputs/phase2/growth_scale.json with {"A_gk_scale": <float>}
Defaults to 1.0 if absent.

Outputs:
  - outputs/phase4/gk_multi.json          # overall stats + per-survey sections + per-bin details
  - (used by companion scripts for table/plots)

WWI logic matches your existing shape-only proxy approach.
"""
import argparse, csv, json, math, os, sys
from pathlib import Path
from collections import defaultdict

def aic(chi2, k, n):
    aic = chi2 + 2*k
    if n - k - 1 > 0:
        aic += (2*k*(k+1)) / (n - k - 1)
    return aic

def bic(chi2, k, n):
    return chi2 + k*math.log(max(n, 1))

def wwi_from_deltas(dA, dB):
    val = 100 * min(1.0, max(0.0, -dA/10.0), max(0.0, -dB/10.0))
    return round(val, 1)

def read_bins(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        need = {"survey","bin_id","z_eff","A_lit","sigma_lit"}
        missing = need - set(r.fieldnames or [])
        if missing:
            print(f"[error] missing columns: {sorted(missing)}", file=sys.stderr); sys.exit(2)
        for i, row in enumerate(r, 2):
            try:
                survey = (row["survey"] or "").strip()
                bin_id = (row["bin_id"] or "").strip()
                z_eff = float(row["z_eff"])
                A = float(row["A_lit"])
                s = float(row["sigma_lit"])
                notes = (row.get("notes") or "").strip()
                if not survey or not bin_id:
                    raise ValueError("empty survey or bin_id")
                if not math.isfinite(z_eff) or not math.isfinite(A) or not math.isfinite(s) or s <= 0:
                    raise ValueError("bad numeric (z_eff/A/sigma) or sigma<=0")
                rows.append(dict(survey=survey, bin_id=bin_id, z_eff=z_eff, A_lit=A, sigma_lit=s, notes=notes))
            except Exception as e:
                print(f"[warn] skip line {i}: {e}", file=sys.stderr)
    if not rows:
        print(f"[error] no valid rows in {csv_path}", file=sys.stderr); sys.exit(2)
    return rows

def weighted_mean(A_sigma_pairs):
    # returns (Ahat, sigma_Ahat, chi2_at_Ahat)
    wsum = 0.0
    vsum = 0.0
    for A, s in A_sigma_pairs:
        w = 1.0/(s*s)
        wsum += w
        vsum += w*A
    Ahat = vsum/wsum
    sA = (1.0/wsum)**0.5
    chi2_best = sum(((A - Ahat)/s)**2 for A, s in A_sigma_pairs)
    return Ahat, sA, chi2_best

def chi2_for_scale(A_sigma_pairs, scale):
    return sum(((A - scale)/s)**2 for A, s in A_sigma_pairs)

def main():
    ap = argparse.ArgumentParser(description="g×κ multi-bin scorer (overall + per-survey)")
    ap.add_argument("--bins", default="data/isw/gk_bins.csv", help="CSV with survey,bin_id,z_eff,A_lit,sigma_lit[,notes]")
    ap.add_argument("--scale", type=float, default=None, help="Predicted shape-only amplitude (A_{gκ})")
    ap.add_argument("--scale-file", default="outputs/phase2/growth_scale.json",
                    help="JSON with {\"A_gk_scale\": <float>} if --scale is not provided")
    ap.add_argument("--out", default="outputs/phase4/gk_multi.json")
    args = ap.parse_args()

    # determine prediction amplitude
    scale = args.scale
    if scale is None:
        try:
            J = json.load(open(args.scale_file))
            scale = float(J.get("A_gk_scale", 1.0))
            print(f"[info] Loaded A_gk_scale={scale} from {args.scale_file}", file=sys.stderr)
        except Exception:
            scale = 1.0
            print("[info] No --scale and no usable scale-file; defaulting A_gk_scale=1.0", file=sys.stderr)

    # read bins
    rows = read_bins(args.bins)
    A0 = 1.0

    # overall stats
    A_sigma_pairs = [(r["A_lit"], r["sigma_lit"]) for r in rows]
    n = len(A_sigma_pairs)
    Ahat, sA, chi2_best = weighted_mean(A_sigma_pairs)
    chi2_pred = chi2_for_scale(A_sigma_pairs, scale)
    chi2_null = chi2_for_scale(A_sigma_pairs, A0)

    k = 1
    AIC_pred = aic(chi2_pred, k, n); AIC_null = aic(chi2_null, k, n)
    BIC_pred = bic(chi2_pred, k, n); BIC_null = bic(chi2_null, k, n)
    dAIC = AIC_pred - AIC_null; dBIC = BIC_pred - BIC_null
    WWI = wwi_from_deltas(dAIC, dBIC)
    dchi_pred = ((Ahat - scale)/sA)**2

    # per-survey aggregates
    by_survey = defaultdict(list)
    for r in rows:
        by_survey[r["survey"]].append(r)

    surveys = {}
    for sname, items in by_survey.items():
        pairs = [(x["A_lit"], x["sigma_lit"]) for x in items]
        ns = len(pairs)
        Ahat_s, sA_s, chi2_best_s = weighted_mean(pairs)
        chi2_pred_s = chi2_for_scale(pairs, scale)
        chi2_null_s = chi2_for_scale(pairs, A0)
        AIC_pred_s = aic(chi2_pred_s, k, ns); AIC_null_s = aic(chi2_null_s, k, ns)
        BIC_pred_s = bic(chi2_pred_s, k, ns); BIC_null_s = bic(chi2_null_s, k, ns)
        dAIC_s = AIC_pred_s - AIC_null_s; dBIC_s = BIC_pred_s - BIC_null_s
        WWI_s = wwi_from_deltas(dAIC_s, dBIC_s)
        dchi_pred_s = ((Ahat_s - scale)/sA_s)**2
        details = []
        for x in items:
            resid_pred = (x["A_lit"] - scale)/x["sigma_lit"]
            resid_null = (x["A_lit"] - A0)/x["sigma_lit"]
            details.append({
                "survey": x["survey"],
                "bin_id": x["bin_id"],
                "z_eff": x["z_eff"],
                "A_lit": x["A_lit"],
                "sigma_lit": x["sigma_lit"],
                "resid_sigma_model": resid_pred,
                "resid_sigma_null": resid_null,
                "notes": x.get("notes","")
            })
        surveys[sname] = {
            "n": ns,
            "A_hat": Ahat_s, "sigma_A_hat": sA_s,
            "chi2_pred": chi2_pred_s, "chi2_null": chi2_null_s, "chi2_best": chi2_best_s,
            "AIC": {"value": AIC_pred_s, "null": AIC_null_s, "delta": dAIC_s},
            "BIC": {"value": BIC_pred_s, "null": BIC_null_s, "delta": dBIC_s},
            "WWI": WWI_s,
            "delta_chi2_pred": dchi_pred_s,
            "details": details
        }

    out = {
        "n": n,
        "scale_model": scale,
        "A_hat": Ahat, "sigma_A_hat": sA, "chi2_best": chi2_best,
        "chi2_pred": chi2_pred, "chi2_null": chi2_null,
        "AIC": {"value": AIC_pred, "null": AIC_null, "delta": dAIC},
        "BIC": {"value": BIC_pred, "null": BIC_null, "delta": dBIC},
        "WWI": WWI,
        "delta_chi2_pred": dchi_pred,
        "surveys": surveys,
        "bins": rows
    }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    mean_sigma = math.sqrt(chi2_pred/max(n,1))
    print(f"Wrote {args.out}  (n={n}, ⟨|res|⟩≈{mean_sigma:.2f}σ, ΔAIC={dAIC:.3f}, WWI={WWI})")

if __name__ == "__main__":
    main()
