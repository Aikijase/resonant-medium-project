# tools/phase4_gk_score.py
#!/usr/bin/env python3
"""
Phase-4: galaxy–kappa (g×κ) shape-only scorer with diagnostics.

Inputs
------
- --catalogs CSV with columns: name,z_eff,A_lit,sigma_lit[,notes]
- --scale     Model's predicted shape-only amplitude A_{gκ} (float)
- --scale-file JSON file with {"A_gk_scale": <float>} to supply --scale
               If both --scale and --scale-file are omitted or invalid, defaults to 1.0.

Output
------
Writes JSON to --out (default: outputs/phase4/gk_score.json) with:
{
  "n": <int>,
  "scale_model": <float>,
  "chi2": <float>,           # χ² for prediction vs literature
  "chi2_null": <float>,      # χ² for LCDM=1.0 vs literature
  "AIC": {"value":..., "null":..., "delta":...},
  "BIC": {"value":..., "null":..., "delta":...},
  "WWI": <float>,            # shape-only win metric (0..100)
  "A_hat": <float>,          # weighted-mean best-fit amplitude from data
  "sigma_A_hat": <float>,    # its uncertainty
  "chi2_best": <float>,      # χ² at A_hat
  "delta_chi2_pred": <float>,# (A_hat - scale_model)^2 / sigma_A_hat^2
  "details": [
     {"name":..., "z_eff":..., "A_lit":..., "sigma_lit":..., 
      "resid_sigma_model":..., "resid_sigma_null":..., "notes": "..."},
     ...
  ]
}
"""
import argparse, csv, json, math, os, sys
from pathlib import Path

def load_catalogs(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, 1):
            try:
                name      = row["name"].strip()
                z_eff     = float(row["z_eff"])
                A_lit     = float(row["A_lit"])
                sigma_lit = float(row["sigma_lit"])
                notes     = row.get("notes","").strip()
                if sigma_lit <= 0:
                    raise ValueError("sigma_lit must be > 0")
                rows.append(dict(name=name, z_eff=z_eff, A_lit=A_lit, sigma_lit=sigma_lit, notes=notes))
            except Exception as e:
                print(f"[warn] Skipping row {i}: {e}", file=sys.stderr)
    return rows

def aic(chi2, k, n):  # small-sample corrected AICc
    aic = chi2 + 2*k
    if n - k - 1 > 0:
        aic += (2*k*(k+1)) / (n - k - 1)
    return aic

def bic(chi2, k, n):
    return chi2 + k*math.log(max(n, 1))

def wwi_from_deltas(dA, dB):
    # Positive deltas => worse than null; negative => better than null.
    # Reward improvements up to 100; neutral if not improving.
    val = 100 * min(1.0,
                    max(0.0, -dA/10.0),
                    max(0.0, -dB/10.0))
    return round(val, 1)

def main():
    ap = argparse.ArgumentParser(description="Phase-4 g×κ (galaxy–kappa) shape-only score with diagnostics")
    ap.add_argument("--catalogs", default="data/isw/gk_catalogs.csv",
                    help="CSV with columns: name,z_eff,A_lit,sigma_lit[,notes]")
    ap.add_argument("--scale", type=float, default=None,
                    help="Model shape-only amplitude A_{gκ} relative to LCDM (default: read from --scale-file or 1.0)")
    ap.add_argument("--scale-file", default="outputs/phase2/growth_scale.json",
                    help="Optional JSON file: {\"A_gk_scale\": <float>} to supply --scale")
    ap.add_argument("--out", default="outputs/phase4/gk_score.json",
                    help="Output JSON path")
    args = ap.parse_args()

    # Determine model scale
    scale = args.scale
    if scale is None:
        try:
            J = json.load(open(args.scale_file))
            scale = float(J.get("A_gk_scale", 1.0))
            print(f"[info] Loaded A_gk_scale={scale} from {args.scale_file}", file=sys.stderr)
        except Exception:
            scale = 1.0
            print("[info] No --scale and no usable scale-file; defaulting A_gk_scale=1.0", file=sys.stderr)

    # Load catalogs
    cats = load_catalogs(args.catalogs)
    if not cats:
        print(f"[error] No valid rows in {args.catalogs}", file=sys.stderr)
        sys.exit(2)

    # Null (LCDM) hypothesis
    A0 = 1.0

    # χ² for prediction and null
    chi2_model = 0.0
    chi2_null  = 0.0
    details = []
    for c in cats:
        A_lit, sig = c["A_lit"], c["sigma_lit"]
        r_model = (A_lit - scale)/sig
        r_null  = (A_lit - A0)/sig
        chi2_model += r_model*r_model
        chi2_null  += r_null*r_null
        details.append({
            "name": c["name"],
            "z_eff": c["z_eff"],
            "A_lit": A_lit,
            "sigma_lit": sig,
            "resid_sigma_model": r_model,
            "resid_sigma_null": r_null,
            "notes": c.get("notes","")
        })

    n = len(cats)
    k_model = 1
    k_null  = 1

    AIC_model = aic(chi2_model, k_model, n)
    AIC_null  = aic(chi2_null,  k_null,  n)
    BIC_model = bic(chi2_model, k_model, n)
    BIC_null  = bic(chi2_null,  k_null,  n)

    dAIC = AIC_model - AIC_null
    dBIC = BIC_model - BIC_null
    WWI  = wwi_from_deltas(dAIC, dBIC)

    # Weighted-mean best-fit amplitude and diagnostics
    wsum = 0.0
    vsum = 0.0
    for c in cats:
        w = 1.0 / (c["sigma_lit"]**2)
        wsum += w
        vsum += w * c["A_lit"]
    A_hat = vsum / wsum
    sigma_hat = (1.0 / wsum) ** 0.5

    chi2_best = 0.0
    for c in cats:
        r_best = (c["A_lit"] - A_hat) / c["sigma_lit"]
        chi2_best += r_best * r_best

    delta_chi2_pred = ((A_hat - scale) / sigma_hat) ** 2

    out = {
        "n": n,
        "scale_model": scale,
        "chi2": chi2_model,
        "chi2_null": chi2_null,
        "AIC": {"value": AIC_model, "null": AIC_null, "delta": dAIC},
        "BIC": {"value": BIC_model, "null": BIC_null, "delta": dBIC},
        "WWI": WWI,
        "A_hat": A_hat,
        "sigma_A_hat": sigma_hat,
        "chi2_best": chi2_best,
        "delta_chi2_pred": delta_chi2_pred,
        "details": details
    }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}  (n={n}, chi2={chi2_model:.3f}, ΔAIC={dAIC:.3f}, WWI={WWI})")

if __name__ == "__main__":
    main()
