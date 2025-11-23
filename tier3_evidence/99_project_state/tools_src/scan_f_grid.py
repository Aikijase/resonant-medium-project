#!/usr/bin/env python3
import subprocess, json, numpy as np, pathlib, csv, sys, time, textwrap

BAO_CSV = "data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV = "data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV  = "data/pantheon_plus/sn_MATCHED_mu.csv"          # unused (we run --bao-only)
SN_COV  = "data/pantheon_plus/SN_MATCHED_sigma2.diag.csv" # unused (we run --bao-only)

OUTDIR  = pathlib.Path("outputs/scan_f_nodamp")
OUTDIR.mkdir(parents=True, exist_ok=True)

# start short; you can expand to np.arange(1.0, 6.0+1e-9, 0.05) once it looks good
FGRID = np.arange(2.50, 3.51, 0.25)  # 2.50, 2.75, 3.00, 3.25, 3.50

def run_one(label, force_A0: bool, scales_mode: str, fval: float):
    """
    scales_mode: 'scales'  (normal two per-kind scales)
                 'pinned'  (emulate 'no scales' by pinning priors tightly at 1.0)
    """
    fstr = f"{fval:.2f}".replace(".", "p")   # dot-free suffix so runner writes foo_f3.json
    outpref = OUTDIR / f"{label}_f{fstr}"

    cmd = [
        sys.executable, "-u", "tools/run_joint_resonant_lock_fixed.py",
        "--bao-csv", BAO_CSV, "--bao-cov", BAO_COV,
        "--sn-csv", SN_CSV, "--sn-cov", SN_COV,
        "--out-prefix", str(outpref),
        "--bao-only",
        "--optimizer", "lbfgsb", "--max-evals", "400",
        "--fix-gamma-zero",
        "--prior-f-mean", f"{fval}", "--prior-f-sigma", "0.03",
    ]
    if scales_mode == "pinned":
        # emulate “no scales”: clamp s_DM, s_DH ~ N(1, σ=1e-9)
        cmd += ["--prior-sdm-sigma", "1e-9", "--prior-sdh-sigma", "1e-9"]
    if force_A0:
        # practically fix A=0 at this f
        cmd += ["--A0", "0.0", "--prior-A-sigma", "1e-9"]

    print(f"[scan] f={fval:.2f} mode={scales_mode:<6} nullA={force_A0}")
    t0=time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt=time.time()-t0

    jpath = str(outpref) + ".json"
    try:
        J = json.load(open(jpath)); ok=True; err=""
    except Exception as e:
        ok=False
        err = (r.stderr or "") + "\n" + (r.stdout or "")
        J = {"chi2": float("nan"), "breakdown": {}, "best_params": {}}

    chi2_bao = float(J.get("breakdown",{}).get("chi2_bao", float("nan")))
    print(f"  -> ok={ok}  chi2_bao={chi2_bao:.6g}  time={dt:.1f}s")
    if not ok:
        print("  ----- STDERR+STDOUT (last 2000 chars) -----")
        print(textwrap.indent(err[-2000:], "  "))
        print("  ------------------------------------------")

    bp = J.get("best_params", {})
    return {
        "label": label, "f": fval, "ok": ok, "chi2_tot": float(J.get("chi2", float("nan"))),
        "chi2_bao": chi2_bao, "A": bp.get("A"), "phi": bp.get("phi"),
        "sDM": bp.get("s_DM"), "sDH": bp.get("s_DH"), "err": "" if ok else "fail",
    }

def main():
    rows=[]
    for f in FGRID:
        rows.append(run_one("scales_best",  False, "scales", f))
        rows.append(run_one("scales_nullA", True,  "scales", f))
        rows.append(run_one("pinned_best",  False, "pinned", f))
        rows.append(run_one("pinned_nullA", True,  "pinned", f))

    csv_path = OUTDIR / "scan_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        cols=["label","f","ok","chi2_tot","chi2_bao","A","phi","sDM","sDH","err"]
        w=csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("Wrote", csv_path)
    print("How to read: per branch, Δχ²_bao = chi2_bao(nullA) - chi2_bao(best).")

if __name__ == "__main__":
    main()
