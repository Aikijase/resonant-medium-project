#!/usr/bin/env python3
"""
Phase-2: Hubble vs Resonant Suppression — κ-only (v4)

- Baseline: LCDM-like growth shape (from column if provided, or crude LCDM).
- Resonant: multiply baseline by S(z; κ) = 1 / (1 + κ * G(a)),
  with G(a) = [a(1-a)]**p and fixed p (default 1.3).
- For each model, fit a single amplitude A* by WLS (focus on shape).
- Params counted: baseline k=1 (A), resonant k=2 (A + κ).  Tau is not used.
"""
import argparse, csv, json, math, os
import numpy as np

def safe_float(x, default=np.nan):
    try: return float(str(x).strip())
    except Exception: return default

def load_fs8_table(path):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            z = safe_float(row.get("z"))
            y = safe_float(row.get("fs8"))
            sig = np.nan
            for nm in ("sigma","err"):
                if nm in row and row[nm] not in (None,""):
                    sig = safe_float(row[nm]); break
            if np.isnan(sig): sig = 0.05
            rows.append({"z":z,"fs8":y,"sigma":sig, **{k:v for k,v in row.items() if k not in ("z","fs8","sigma","err")}})
    rows=[d for d in rows if np.isfinite(d["z"]) and np.isfinite(d["fs8"]) and np.isfinite(d["sigma"]) and d["sigma"]>0]
    if not rows: raise RuntimeError("No valid rows in fs8 CSV.")
    return rows

def lcdm_curve(z, Om0):
    a=1.0/(1.0+z)
    Omz=Om0/(Om0+(1-Om0)*a**3)
    return 0.5*(Omz**0.55)

def get_baseline_series(data, Om0, baseline_col):
    if baseline_col and (baseline_col in data[0]):
        base=np.array([safe_float(d[baseline_col]) for d in data])
        bad=~np.isfinite(base)
        if bad.any():
            base[bad]=np.array([lcdm_curve(d["z"], Om0) if b else base[i]
                                for i,(d,b) in enumerate(zip(data,bad))])
    else:
        base=np.array([lcdm_curve(d["z"], Om0) for d in data])
    return base

def G_kernel(a, p):
    return (a*(1.0-a))**p

def suppression(a, kappa, p):
    return 1.0/(1.0 + kappa*G_kernel(a,p))

def model_shape_series(data, base, kappa, p):
    a=np.array([1.0/(1.0+d["z"]) for d in data])
    return base * suppression(a, kappa, p)

def fit_A_WLS(y, m, s):
    w=1.0/np.square(s); num=(w*y*m).sum(); den=(w*np.square(m)).sum()
    return (num/den) if den>0 else 1.0

def chi2(y, m, s): return float(np.sum(np.square((y-m)/s)))
def aic(c2,k): return c2 + 2*k
def bic(c2,k,n): return c2 + k*math.log(n)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fs8-data", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--Om0", type=float, default=0.30)
    ap.add_argument("--baseline-col", default=None)
    ap.add_argument("--p", type=float, default=1.3, help="fixed kernel exponent for G(a)=[a(1-a)]^p")
    ap.add_argument("--kappa-range", nargs=3, type=float, default=[0.00,0.30,16], help="start stop steps")
    args=ap.parse_args()

    data=load_fs8_table(args.fs8_data); n=len(data)
    y=np.array([d["fs8"] for d in data]); s=np.array([d["sigma"] for d in data])
    base=get_baseline_series(data, args.Om0, args.baseline_col)

    # Baseline (A only)
    A0=fit_A_WLS(y, base, s); y0=A0*base
    chi0=chi2(y,y0,s); AIC0=aic(chi0,1); BIC0=bic(chi0,1,n)

    k0,k1,kn=args.kappa_range
    kappas=np.linspace(k0,k1,int(kn))

    rows=[]; best={"chi2":1e99}
    for kap in kappas:
        m=model_shape_series(data, base, kap, args.p)
        A=fit_A_WLS(y, m, s); yhat=A*m
        c2=chi2(y,yhat,s)
        AIC=aic(c2,2); BIC=bic(c2,2,n)  # k=2 (A + kappa)
        rows.append({
            "kappa": float(kap), "A_amp": float(A),
            "chi2": float(c2), "AIC": float(AIC), "BIC": float(BIC),
            "dAIC_vs_LCDM": float(AIC-AIC0), "dBIC_vs_LCDM": float(BIC-BIC0)
        })
        if c2 < best["chi2"]:
            best={"kappa":float(kap),"A_amp":float(A),"chi2":float(c2),
                  "AIC":float(AIC),"BIC":float(BIC),
                  "dAIC":float(AIC-AIC0),"dBIC":float(BIC-BIC0)}

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    # CSV
    import csv as _csv
    with open(f"{args.out_prefix}.csv","w",newline="") as f:
        w=_csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # TXT
    with open(f"{args.out_prefix}.txt","w") as f:
        f.write("=== Phase-2: κ-only resonant suppression (v4) ===\n")
        f.write(f"fs8 file: {args.fs8_data}\n")
        if args.baseline_col: f.write(f"baseline_col: {args.baseline_col}\n")
        f.write(f"n={n}  p={args.p}\n")
        f.write(f"Baseline: A*={A0:.6f}  chi2={chi0:.3f}  AIC={AIC0:.3f}  BIC={BIC0:.3f}\n\n")
        f.write(f"BEST: kappa={best['kappa']:.6f}  A_amp={best['A_amp']:.6f}  chi2={best['chi2']:.3f}  "
                f"dAIC={best['dAIC']:.3f}  dBIC={best['dBIC']:.3f}\n")
        verdict = "PASS" if (best["dAIC"]<0 and best["dBIC"]<=0) else "TIE/FAIL"
        f.write(f"VERDICT: {verdict}\n")
    # JSON
    with open(f"{args.out_prefix}.json","w") as f:
        json.dump({
            "n": n, "p": args.p,
            "baseline": {"A": A0, "chi2": chi0, "AIC": AIC0, "BIC": BIC0},
            "best": best
        }, f, indent=2)

    print(f"[v4] Baseline chi2={chi0:.3f}  AIC={AIC0:.3f}  BIC={BIC0:.3f} (A*={A0:.6f})")
    print(f"[v4] Best  kappa={best['kappa']:.4f}  chi2={best['chi2']:.3f}  ΔAIC={best['dAIC']:.3f}  ΔBIC={best['dBIC']:.3f}")
    print(f"[v4] VERDICT: {'PASS' if (best['dAIC']<0 and best['dBIC']<=0) else 'TIE/FAIL'}")
