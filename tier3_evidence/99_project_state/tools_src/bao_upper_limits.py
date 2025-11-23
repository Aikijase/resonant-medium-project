#!/usr/bin/env python3
import subprocess, json, numpy as np, pathlib, csv, sys, time, textwrap

BAO_CSV = "data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV = "data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV  = "data/pantheon_plus/sn_MATCHED_mu.csv"          # unused (bao-only)
SN_COV  = "data/pantheon_plus/SN_MATCHED_sigma2.diag.csv" # unused (bao-only)

OUTDIR  = pathlib.Path("outputs/bao_limits")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Short grid first; expand to np.arange(1.0, 6.0+1e-9, 0.05) after it works
FGRID = np.arange(2.0, 3.51, 0.25)

CHI2_TARGET = 3.84   # ~95% CL for 1 dof
TIMEOUT_S   = 120    # per subprocess

def run_fit(label, fval, A0=None, clamp_A_sigma=None,
            sdm_sigma=0.05, sdh_sigma=0.05, maxevals=400, verbose=False):
    fstr = f"{fval:.2f}".replace(".", "p")
    outpref = OUTDIR / f"{label}_f{fstr}"
    cmd = [
        sys.executable, "-u", "tools/run_joint_resonant_lock_fixed.py",
        "--bao-csv", BAO_CSV, "--bao-cov", BAO_COV,
        "--sn-csv", SN_CSV, "--sn-cov", SN_COV,
        "--out-prefix", str(outpref),
        "--bao-only",
        "--optimizer", "lbfgsb", "--max-evals", str(maxevals),
        "--fix-gamma-zero",
        "--prior-f-mean", f"{fval}", "--prior-f-sigma", "0.03",
        "--prior-sdm-sigma", f"{sdm_sigma}", "--prior-sdh-sigma", f"{sdh_sigma}",
    ]
    if A0 is not None and clamp_A_sigma is not None:
        cmd += ["--A0", f"{A0}", "--prior-A-sigma", f"{clamp_A_sigma}"]

    if verbose:
        print("  [fit] Running:", " ".join(cmd))
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        if verbose: print("  [fit] TIMEOUT after", TIMEOUT_S, "s")
        return False, float("nan"), {}
    dt = time.time() - t0

    jpath = str(outpref) + ".json"
    try:
        J = json.load(open(jpath))
        ok = bool(J.get("success", False))
        chi2_bao = float(J.get("breakdown", {}).get("chi2_bao", float("nan")))
        if verbose:
            print(f"  [fit] ok={ok} chi2_bao={chi2_bao:.6g} t={dt:.1f}s -> {jpath}")
        return ok, chi2_bao, J.get("best_params", {})
    except Exception:
        if verbose:
            tail = (r.stderr or "")[-400:]
            print("  [fit] FAILED JSON. stderr tail:", tail)
        return False, float("nan"), {}

def bracket_A95(f, chi2_best, target, sdm_sigma=0.05, sdh_sigma=0.05):
    def chi2_at(A, tag):
        print(f"    [bracket] A={A:.5f} ({tag})")
        ok, chi2A, _ = run_fit("testA", f, A0=A, clamp_A_sigma=1e-9,
                               sdm_sigma=sdm_sigma, sdh_sigma=sdh_sigma,
                               maxevals=250, verbose=False)
        if not ok or not np.isfinite(chi2A):
            print("      -> bad chi2; treating as +inf")
            return float("inf")
        print(f"      -> chi2={chi2A:.6g}  Δ={chi2A-chi2_best:.6g}")
        return chi2A

    lo, hi = 0.0, 0.1
    _ = chi2_at(lo, "lo")
    tries = 0
    while True:
        chi2_hi = chi2_at(hi, "hi")
        if chi2_hi >= target: break
        hi *= 1.6
        tries += 1
        if tries > 12 or hi > 5.0:
            print("    [bracket] Could not bracket target; giving up.")
            return float("nan")

    for _ in range(28):
        mid = 0.5*(lo+hi)
        chi2_mid = chi2_at(mid, "mid")
        if not np.isfinite(chi2_mid):
            lo = mid; continue
        if chi2_mid >= target: hi = mid
        else: lo = mid
        if hi - lo < 1e-3: break
    return hi

def main():
    rows = []
    for f in FGRID:
        print(f"\n=== f = {f:.2f} ===")
        okb, chi2_best, bp  = run_fit("best",  f, verbose=True)
        ok0, chi2_null, _   = run_fit("nullA", f, A0=0.0, clamp_A_sigma=1e-9, verbose=True)
        if not (okb and ok0):
            print("  [warn] best/nullA failed; skipping f")
            rows.append({"f": f, "ok": False, "msg": "fit failed",
                         "A_best": None, "chi2_bao_best": float("nan"),
                         "chi2_bao_null": float("nan"), "dchi2": float("nan"),
                         "A95": float("nan")})
            continue
        dchi2 = chi2_null - chi2_best
        target = chi2_best + CHI2_TARGET
        A_best = bp.get("A", None)
        print(f"  chi2_best={chi2_best:.6g}  chi2_null={chi2_null:.6g}  dchi2={dchi2:.6g}")
        A95 = bracket_A95(f, chi2_best, target)
        rows.append({"f": f, "ok": True, "msg": "",
                     "A_best": A_best, "chi2_bao_best": chi2_best,
                     "chi2_bao_null": chi2_null, "dchi2": dchi2, "A95": A95})

    out = OUTDIR / "bao_upper_limits.csv"
    with open(out, "w", newline="") as fh:
        cols = ["f","ok","msg","A_best","chi2_bao_best","chi2_bao_null","dchi2","A95"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("Wrote", out)
    print("Columns: f, A_best, dchi2 (null-best), A95 (95% CL upper limit).")

if __name__ == "__main__":
    main()
