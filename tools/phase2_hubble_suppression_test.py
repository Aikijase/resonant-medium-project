#!/usr/bin/env python3
"""
Phase-2: Hubble vs Resonant Suppression Test (v2, with amplitude nuisance)
Compares LCDM-like (Hubble-only) vs Resonant (Hubble + κ, optional τ),
NOW fitting a per-grid multiplicative amplitude A* in closed-form (WLS).
Outputs: CSV + TXT + JSON with chi2/AIC/BIC/WWI and a verdict.
If your fs8 CSV includes a baseline prediction column (e.g. fs8_lcdm_repaired),
set --baseline-col to use it instead of the toy LCDM curve.
"""
import argparse, json, math, csv, os
import numpy as np

def load_fs8_table(path):
    rows=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            z=float(row["z"])
            fs8=float(row["fs8"])
            sig=float(row.get("sigma", row.get("err", "0.05")) or 0.05)
            rows.append({"z":z,"fs8":fs8,"sigma":sig, **row})
    return rows

def lcdm_curve(z, Om0):
    # very crude stand-in; prefer baseline-col if available
    a=1.0/(1.0+z)
    Omz = Om0 / (Om0 + (1-Om0)*a**3)
    return 0.5*(Omz**0.55)

def get_baseline_series(data, Om0, baseline_col=None):
    if baseline_col and baseline_col in data[0]:
        # Use provided repaired LCDM if available
        base=[float(d[baseline_col]) for d in data]
    else:
        base=[lcdm_curve(d["z"], Om0) for d in data]
    return np.array(base)

def resonant_suppression_factor(z, kappa, tau):
    # Smooth mid-z kernel: peaks near z~1 (a~0.5), widens mildly with τ
    a=1.0/(1.0+z)
    G = a*(1-a)*(1+0.5*tau)
    return 1.0/(1.0 + kappa*G)

def model_series(data, base_series, kappa, tau):
    fac=np.array([resonant_suppression_factor(d["z"], kappa, tau) for d in data])
    return base_series*fac

def fit_amplitude_WLS(y, m, sig):
    # Min_A sum((y - A m)^2 / sig^2) => A* = sum(w y m)/sum(w m^2)
    w = 1.0/np.square(sig)
    num = np.sum(w*y*m)
    den = np.sum(w*np.square(m))
    A = num/den if den>0 else 1.0
    return A

def chi2(y, m, sig):
    return float(np.sum(np.square((y-m)/sig)))

def aic(chi2_val, k): return chi2_val + 2*k
def bic(chi2_val, k, n): return chi2_val + k*math.log(n)
def wwi_from_deltas(dA, dB): return 100*min(1.0, max(0,-dA/10.0), max(0,-dB/10.0))

def run_suite(fs8_csv, out_prefix, kappa_grid, tau_grid, Om0, baseline_col=None):
    data = load_fs8_table(fs8_csv)
    n=len(data)
    y  = np.array([d["fs8"] for d in data])
    s  = np.array([d["sigma"] for d in data])
    base = get_baseline_series(data, Om0, baseline_col)

    # Baseline (Hubble-only) + fitted amplitude
    m0_shape = base
    A0 = fit_amplitude_WLS(y, m0_shape, s)
    y0 = A0*m0_shape
    chi2_0 = chi2(y, y0, s)
    k0 = 1  # amplitude A0 as 1 param (Om0 fixed here)
    AIC0 = aic(chi2_0, k0); BIC0 = bic(chi2_0, k0, n)

    rows=[]; best={"chi2":1e99}
    for kap in kappa_grid:
        for tau in tau_grid:
            m_shape = model_series(data, base, kap, tau)
            A = fit_amplitude_WLS(y, m_shape, s)
            yhat = A*m_shape
            c2 = chi2(y, yhat, s)
            # params: A + (kappa>0 ? kappa : 0) + (tau>0 ? tau : 0)
            k = 1 + (1 if kap!=0 else 0) + (1 if tau!=0 else 0)
            AIC = aic(c2, k); BIC = bic(c2, k, n)
            dA = AIC - AIC0; dB = BIC - BIC0
            WWI = wwi_from_deltas(dA, dB)
            row={"kappa":kap,"tau":tau,"A_amp":A,"chi2":c2,"AIC":AIC,"BIC":BIC,
                 "dAIC_vs_LCDM":dA,"dBIC_vs_LCDM":dB,"WWI":WWI}
            rows.append(row)
            if c2 < best["chi2"]: best=row

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    # CSV
    csv_path=f"{out_prefix}.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # TXT
    txt_path=f"{out_prefix}.txt"
    with open(txt_path,"w") as f:
        f.write("=== Phase-2: Hubble vs Resonant Suppression Test (v2) ===\n")
        f.write(f"fs8 file: {fs8_csv}\n")
        if baseline_col: f.write(f"baseline_col: {baseline_col}\n")
        f.write(f"n = {n}\n")
        f.write(f"Baseline (LCDM-like) A*={A0:.6f}  chi2={chi2_0:.3f}  AIC={AIC0:.3f}  BIC={BIC0:.3f}\n")
        f.write("\n--- Best Resonant (after amplitude fit) ---\n")
        for k in ["kappa","tau","A_amp","chi2","AIC","BIC","dAIC_vs_LCDM","dBIC_vs_LCDM","WWI"]:
            v=best[k]; f.write(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}"); f.write("\n")
        verdict = "PASS" if (best["AIC"] < AIC0 and best["BIC"] <= BIC0) else "TIE/FAIL"
        f.write(f"\nVERDICT: {verdict}\n")
    # JSON
    json_path=f"{out_prefix}.json"
    with open(json_path,"w") as f:
        json.dump({"fs8_csv":fs8_csv,"n":n,"baseline":{"A":A0,"chi2":chi2_0,"AIC":AIC0,"BIC":BIC0},
                   "best":best,"baseline_col":baseline_col}, f, indent=2)
    return csv_path, txt_path, json_path

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fs8-data", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--Om0", type=float, default=0.30)
    p.add_argument("--kappa-range", nargs=3, type=float, default=[0.00,0.25,11])
    p.add_argument("--tau-range",   nargs=3, type=float, default=[0.00,0.40, 9])
    p.add_argument("--baseline-col", default=None,
                   help="Column name in CSV that holds baseline LCDM fs8 (e.g. fs8_lcdm_repaired).")
    args=p.parse_args()
    import numpy as np
    k0,k1,kn = args.kappa_range
    t0,t1,tn = args.tau_range
    kappa_grid=np.linspace(k0,k1,int(kn)).tolist()
    tau_grid=np.linspace(t0,t1,int(tn)).tolist()
    run_suite(args.fs8_data, args.out_prefix, kappa_grid, tau_grid, args.Om0, args.baseline_col)
