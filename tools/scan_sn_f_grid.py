#!/usr/bin/env python3
import subprocess, json, numpy as np, pathlib, csv, sys, time, textwrap

SN_CSV="data/pantheon_plus/sn_MATCHED_mu.csv"
SN_COV="data/pantheon_plus/SN_MATCHED_sigma2.diag.csv"
BAO_CSV="data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV="data/desi_dr1_bao/bao_covariance_plus_lya.csv"

OUTDIR=pathlib.Path("outputs/scan_sn_nodamp")
OUTDIR.mkdir(parents=True, exist_ok=True)

# frequency grid
FGRID = np.arange(2.0, 5.0+1e-9, 0.10)

def run_one(label, force_A0, fval):
    fstr=f"{fval:.2f}".replace(".","p")
    outpref=OUTDIR/f"{label}_f{fstr}"
    cmd=[sys.executable,"-u","tools/run_joint_resonant_lock_fixed.py",
         "--bao-csv",BAO_CSV,"--bao-cov",BAO_COV,
         "--sn-csv",SN_CSV,"--sn-cov",SN_COV,
         "--out-prefix",str(outpref),
         "--sn-only","--optimizer","lbfgsb","--max-evals","400",
         "--fix-gamma-zero","--prior-f-mean",f"{fval}","--prior-f-sigma","0.03"]
    if force_A0:
        cmd+=["--A0","0.0","--prior-A-sigma","1e-9"]

    print(f"[SN-scan] f={fval:.2f} nullA={force_A0}")
    t0=time.time()
    r=subprocess.run(cmd,capture_output=True,text=True)
    dt=time.time()-t0
    jpath=str(outpref)+".json"
    try:
        J=json.load(open(jpath)); ok=True; err=""
    except Exception as e:
        ok=False; err=(r.stderr or "")+"\n"+(r.stdout or "")
        J={"breakdown":{},"best_params":{}}

    chi2_sn=float(J.get("breakdown",{}).get("chi2_sn",float("nan")))
    print(f"  -> ok={ok}  chi2_sn={chi2_sn:.6g}  time={dt:.1f}s")
    if not ok:
        print("  --- LOG (tail) ---")
        print(textwrap.indent(err[-1500:], "  "))

    bp=J.get("best_params",{})
    return {"label":label,"f":fval,"ok":ok,"chi2_sn":chi2_sn,
            "A":bp.get("A"),"phi":bp.get("phi")}

def main():
    rows=[]
    for f in FGRID:
        rows.append(run_one("sn_best", False, f))
        rows.append(run_one("sn_nullA", True, f))
    csv_path=OUTDIR/"scan_sn_summary.csv"
    with open(csv_path,"w",newline="") as fh:
        cols=["label","f","ok","chi2_sn","A","phi"]
        w=csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("Wrote", csv_path)
    print("How to read: Δχ²_sn = chi2_sn(nullA) - chi2_sn(best).")

if __name__=="__main__":
    main()
