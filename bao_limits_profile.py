cat > tools/bao_limits_profile.py <<'PY'
#!/usr/bin/env python3
import subprocess, json, numpy as np, pathlib, csv, sys, time

# Paths
BAO_CSV = "data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV = "data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV  = "data/pantheon_plus/sn_MATCHED_mu.csv"
SN_COV  = "data/pantheon_plus/SN_MATCHED_sigma2.diag.csv"

OUTDIR = pathlib.Path("outputs/bao_limits_profile")
OUTDIR.mkdir(parents=True, exist_ok=True)

FGRID = np.arange(2.0, 3.51, 0.25)
CHI2_95 = 3.84  # 95% CL, 1 d.o.f.
TIMEOUT = 120   # seconds per run

def run_fit(out_prefix, f, A0=None, prior_A_sigma=None, maxevals=400):
    """Run one BAO-only fit and return (ok, chi2_bao, best_params)."""
    cmd = [
        sys.executable, "-u", "tools/run_joint_resonant_lock_fixed.py",
        "--bao-csv", BAO_CSV, "--bao-cov", BAO_COV,
        "--sn-csv", SN_CSV, "--sn-cov", SN_COV,
        "--bao-only",
        "--optimizer", "lbfgsb",
        "--max-evals", str(maxevals),
        "--fix-gamma-zero",
        "--prior-f-mean", str(f),
        "--prior-f-sigma", "0.03"
    ]
    if A0 is not None and prior_A_sigma is not None:
        cmd += ["--A0", str(A0), "--prior-A-sigma", str(prior_A_sigma)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        J = json.load(open(f"{out_prefix}.json"))
        chi2 = float(J["breakdown"]["chi2_bao"])
        return J.get("success", False), chi2, J["best_params"]
    except Exception:
        return False, float("nan"), {}

def chi2_at(f, A):
    """Return chi2_bao for fixed amplitude A."""
    tag = f"A{A:.6f}".replace(".", "p")
    out = OUTDIR / f"{tag}_f{str(f).replace('.', 'p')}"
    ok, chi2, _ = run_fit(out, f, A0=A, prior_A_sigma=1e-9)
    return chi2 if ok and np.isfinite(chi2) else float("inf")

def find_crossing_low(f, chi2_best, target):
    lo, hi = 0.0, 1e-4
    chi2_lo = chi2_at(f, lo)
    if chi2_lo < target:
        return float("nan")
    for _ in range(25):
        chi2_hi = chi2_at(f, hi)
        if chi2_hi <= target:
            break
        hi *= 2
        if hi > 0.1:
            return float("nan")
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        c2 = chi2_at(f, mid)
        if c2 <= target: hi = mid
        else: lo = mid
        if hi - lo < 1e-5: break
    return hi

def find_crossing_high(f, chi2_best, target, A_best):
    lo = max(A_best, 1e-4)
    c2_lo = chi2_at(f, lo)
    if c2_lo >= target:
        return lo
    hi = max(2.0 * lo, 1e-3)
    for _ in range(25):
        c2_hi = chi2_at(f, hi)
        if c2_hi >= target:
            break
        hi *= 1.5
        if hi > 5.0:
            return float("nan")
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        c2 = chi2_at(f, mid)
        if c2 >= target: hi = mid
        else: lo = mid
        if hi - lo < 1e-4: break
    return hi

def main():
    rows = []
    for f in FGRID:
        tag = f"{f:.2f}".replace(".", "p")
        print(f"\n=== f = {f:.2f} ===", flush=True)

        ok_best, chi2_best, bp = run_fit(OUTDIR / f"best_f{tag}", f)
        if not ok_best:
            print("  best fit failed, skipping")
            continue

        ok_null, chi2_null, _ = run_fit(OUTDIR / f"nullA_f{tag}", f, A0=0.0, prior_A_sigma=1e-9)
        if not ok_null:
            print("  null fit failed, skipping")
            continue

        dchi2 = chi2_null - chi2_best
        target = chi2_best + CHI2_95
        A_best = bp.get("A", None)

        print(f"  chi2_best={chi2_best:.3f}  chi2_null={chi2_null:.3f}  Δχ²={dchi2:.3f}  A_best≈{A_best}")

        A_low = find_crossing_low(f, chi2_best, target)
        A_hi = find_crossing_high(f, chi2_best, target, A_best)
        print(f"  95% crossings: A_low≈{A_low}  A_hi≈{A_hi}")

        rows.append({
            "f": f, "A_best": A_best,
            "chi2_best": chi2_best, "chi2_null": chi2_null, "dchi2": dchi2,
            "A_low95": A_low, "A_hi95": A_hi
        })

    out = OUTDIR / "bao_limits_profile.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print("\nWrote:", out)

if __name__ == "__main__":
    main()
PY
