#!/usr/bin/env python3
"""
BAO hBAO Uncertainty v2 — full report (previous data only)

Runs:
  • Ultrafine sweep (best hBAO)
  • Wide sweep (profile 1σ with Δχ²=1, precision CSV)
  • Quadratic 1σ from curvature near the min
  • Baseline vs scaled Δχ² at anchors
  • LOOCV robustness (auto-diag from sigma)
  • H0 sensitivity
Writes:
  - hbao_ultrafine_summary.csv
  - hbao_wide_summary.csv
  - loocv_summary.csv
  - h0_sensitivity_summary.csv
  - bao_hbao_uncertainty_report.txt
"""

import argparse, os, sys, subprocess, json
from pathlib import Path
import numpy as np, pandas as pd

ENGINE = Path("tools/bao_tracer_sweep.py")

def sh(cmd):
    env = os.environ.copy(); env.setdefault("PYTHONPATH",".")
    cp = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if cp.returncode != 0:
        print(cp.stdout); print(cp.stderr)
        raise RuntimeError("Command failed: "+" ".join(cmd))
    return cp.stdout

def parse_best(stdout:str):
    out={}
    for ln in stdout.splitlines():
        if ln.startswith("Best:"):
            for tok in ln.split():
                if "=" in tok:
                    k,v = tok.split("=",1)
                    try: out[k]=float(v)
                    except: pass
            break
    return out

def sweep_ultrafine(a):
    pref = f"{a.out_dir}/hbao_ultrafine"
    cmd = [sys.executable, str(ENGINE),
        "--bao-csv", a.bao_csv, "--cov-csv", a.cov_csv,
        "--om0", str(a.om0), "--h", str(a.h),
        "--z-col", a.z_col, "--y-col", a.y_col, "--kind-col", a.kind_col,
        "--hELG-range", str(a.hbao_min), str(a.hbao_max), str(a.hbao_steps),
        "--hLRG2-range","1.00","1.00","1",
        "--elg-pattern",".*","--lrg2-pattern","$^",
        "--out-prefix", pref
    ]
    out = sh(cmd); best=parse_best(out)
    return best, Path(pref+"_summary.csv")

def sweep_wide(a):
    pref = f"{a.out_dir}/hbao_wide"
    cmd = [sys.executable, str(ENGINE),
        "--bao-csv", a.bao_csv, "--cov-csv", a.cov_csv,
        "--om0", str(a.om0), "--h", str(a.h),
        "--z-col", a.z_col, "--y-col", a.y_col, "--kind-col", a.kind_col,
        "--hELG-range", str(a.hbao_wide_min), str(a.hbao_wide_max), str(a.hbao_wide_steps),
        "--hLRG2-range","1.00","1.00","1",
        "--elg-pattern",".*","--lrg2-pattern","$^",
        "--out-prefix", pref
    ]
    # ensure high-precision CSV in engine
    # (assumes you patched; if not, result is fine but profile σ may be 0)
    out = sh(cmd); best=parse_best(out)
    return best, Path(pref+"_summary.csv")

def quad_sigma(csv_path:Path):
    df = pd.read_csv(csv_path)
    i0 = df['chi2'].idxmin(); h0=df.loc[i0,'hELG']; chi0=df.loc[i0,'chi2']
    win = (df['hELG'] > h0-0.004) & (df['hELG'] < h0+0.004)
    d = df[win].copy()
    x = (d['hELG'] - h0).values; y = d['chi2'].values
    A,B,C = np.polyfit(x, y, 2)
    sig = np.sqrt(1.0/A) if A>0 else np.nan
    return float(h0), float(chi0), float(sig), float(A)

def profile_sigma(csv_path:Path):
    df = pd.read_csv(csv_path)
    i0 = df['chi2'].idxmin(); h0=df.loc[i0,'hELG']; chi0=df.loc[i0,'chi2']
    m = df['chi2'] <= chi0 + 1.0
    lo = df.loc[m & (df['hELG']<=h0),'hELG'].max()
    hi = df.loc[m & (df['hELG']>=h0),'hELG'].min()
    lo = float(lo) if pd.notnull(lo) else h0
    hi = float(hi) if pd.notnull(hi) else h0
    return float(h0), float(chi0), float(h0-lo), float(hi-h0)

def baseline_scaled_delta(a, hbao):
    # baseline
    prefB = f"{a.out_dir}/global_baseline"
    out = sh([sys.executable, str(ENGINE),
        "--bao-csv", a.bao_csv, "--cov-csv", a.cov_csv,
        "--om0", str(a.om0), "--h", str(a.h),
        "--z-col", a.z_col, "--y-col", a.y_col, "--kind-col", a.kind_col,
        "--hELG-fixed","1.00","--hLRG2-fixed","1.00",
        "--elg-pattern","$^","--lrg2-pattern","$^",
        "--out-prefix", prefB])
    b=parse_best(out).get("chi2")

    # scaled
    prefS = f"{a.out_dir}/global_scaled"
    out = sh([sys.executable, str(ENGINE),
        "--bao-csv", a.bao_csv, "--cov-csv", a.cov_csv,
        "--om0", str(a.om0), "--h", str(a.h),
        "--z-col", a.z_col, "--y-col", a.y_col, "--kind-col", a.kind_col,
        "--hELG-fixed", f"{hbao}", "--hLRG2-fixed","1.00",
        "--elg-pattern",".*","--lrg2-pattern","$^",
        "--out-prefix", prefS])
    s=parse_best(out).get("chi2")
    return float(b), float(s), float(b-s)

def loocv(a):
    src = Path(a.bao_csv)
    rows = sum(1 for _ in open(src)) - 1
    rec=[]
    for i in range(1, rows+1):
        tmp = Path(f"/tmp/bao_loocv_{i}.csv")
        with open(src) as fin, open(tmp,"w") as fout:
            header=fin.readline(); fout.write(header)
            for j,line in enumerate(fin,1):
                if j==i: continue
                fout.write(line)
        pref=f"{a.out_dir}/loocv_{i}"
        out = sh([sys.executable,str(ENGINE),
            "--bao-csv",str(tmp),"--cov-csv","auto",
            "--om0",str(a.om0),"--h",str(a.h),
            "--z-col",a.z_col,"--y-col",a.y_col,"--kind-col",a.kind_col,
            "--hELG-range",str(a.hbao_min),str(a.hbao_max),"41",
            "--hLRG2-range","1.00","1.00","1",
            "--elg-pattern",".*","--lrg2-pattern","$^",
            "--out-prefix",pref])
        best=parse_best(out)
        rec.append({"drop_row":i,"hBAO":best.get("hELG"),"chi2":best.get("chi2")})
    df=pd.DataFrame(rec); df.to_csv(f"{a.out_dir}/loocv_summary.csv",index=False)
    return df

def h0_sense(a, hbao):
    rec=[]
    for HH in a.h0_list:
        pref=f"{a.out_dir}/hbao_h{str(HH).replace('.','')}"
        out = sh([sys.executable,str(ENGINE),
            "--bao-csv",a.bao_csv,"--cov-csv",a.cov_csv,
            "--om0",str(a.om0),"--h",str(HH),
            "--z-col",a.z_col,"--y-col",a.y_col,"--kind-col",a.kind_col,
            "--hELG-fixed",str(hbao),"--hLRG2-fixed","1.00",
            "--elg-pattern",".*","--lrg2-pattern","$^",
            "--out-prefix",pref])
        best=parse_best(out)
        rec.append({"H0":HH,"chi2":best.get("chi2")})
    df=pd.DataFrame(rec); df.to_csv(f"{a.out_dir}/h0_sensitivity_summary.csv",index=False)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bao-csv", default="data/desi_dr1_bao/bao_measurements_long_sigma.csv")
    ap.add_argument("--cov-csv", default="data/desi_dr1_bao/bao_covariance.csv")
    ap.add_argument("--om0", type=float, default=0.320)
    ap.add_argument("--h",   type=float, default=0.70)
    ap.add_argument("--z-col", default="z"); ap.add_argument("--y-col", default="y_data"); ap.add_argument("--kind-col", default="kind")
    ap.add_argument("--hbao-min", type=float, default=0.964); ap.add_argument("--hbao-max", type=float, default=0.976); ap.add_argument("--hbao-steps", type=int, default=121)
    ap.add_argument("--hbao-wide-min", type=float, default=0.94); ap.add_argument("--hbao-wide-max", type=float, default=1.00); ap.add_argument("--hbao-wide-steps", type=int, default=241)
    ap.add_argument("--h0-list", nargs="*", type=float, default=[0.68,0.70,0.72])
    ap.add_argument("--out-dir", default="outputs/thrace_best")
    a=ap.parse_args()
    Path(a.out_dir).mkdir(parents=True,exist_ok=True)

    best_u, csv_u = sweep_ultrafine(a)
    h0_u, chi0_u, sig_q, A = quad_sigma(csv_u)

    best_w, csv_w = sweep_wide(a)
    h0_w, chi0_w, sig_lo, sig_hi = profile_sigma(csv_w)

    bchi, schi, dchi = baseline_scaled_delta(a, h0_u)

    df_loocv = loocv(a)
    df_h0    = h0_sense(a, h0_u)

    report = Path(a.out_dir)/"bao_hbao_uncertainty_report.txt"
    with open(report,"w") as f:
        f.write("DESI DR1 BAO — Global hBAO Uncertainty & Robustness (previous data only)\n\n")
        f.write(f"Anchors: Om0={a.om0:.3f}, h={a.h:.2f}\n")
        f.write(f"Ultrafine best: hBAO={h0_u:.6f}, chi2={chi0_u:.6f}\n")
        f.write(f"Baseline vs Scaled at anchors: chi2_baseline={bchi:.6f}, chi2_scaled={schi:.6f}, Δchi2={dchi:.6f}\n\n")
        f.write("Uncertainty:\n")
        f.write(f"  Quadratic 1σ: hBAO = {h0_u:.6f} ± {sig_q:.6f}  (A={A:.3e})\n")
        f.write(f"  Profile  1σ: hBAO = {h0_w:.6f} (-{sig_lo:.6f}, +{sig_hi:.6f}) from wide scan\n\n")
        f.write("LOOCV (drop-one rows):\n"); f.write(df_loocv.to_string(index=False)+"\n\n")
        f.write("H0 sensitivity (fixed best hBAO):\n"); f.write(df_h0.to_string(index=False)+"\n")
    print(f"[write] {report}")

    meta = {
        "Omega_m": a.om0, "h0": a.h,
        "hBAO_best": h0_u, "chi2_min": chi0_u,
        "chi2_baseline": bchi, "delta_chi2": dchi,
        "sigma_quad": sig_q, "sigma_profile_lo": sig_lo, "sigma_profile_hi": sig_hi
    }
    Path(a.out_dir,"hbao_sigma_v2.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {a.out_dir}/hbao_sigma_v2.json")

if __name__=="__main__":
    main()
