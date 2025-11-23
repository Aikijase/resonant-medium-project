#!/usr/bin/env python3
"""
BAO amplitude limits vs f (simple + robust)

For each f:
  • FREE run (A floats) -> A_best, χ²_best (BAO & TOTAL)
  • FIXED A=0           -> χ²_null (BAO)
  • Analytic A95↑ (BAO): A_best + sqrt(3.84)*|A_best|/sqrt(χ²_null_BAO - χ²_best_BAO)
  • Profile A95↑ (TOTAL): fix A and bisect until χ²_total(A)-χ²_best_total = +3.84

Outputs: outputs/bao_limits_profile/bao_limits_profile.csv
"""

from __future__ import annotations
import argparse, csv, json, math, os, re, sys, pathlib, subprocess as sp
import numpy as np

# ---- data paths (edit if needed) ----
BAO_CSV = "data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV = "data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV  = "data/pantheon_plus/sn_MATCHED_mu.csv"
SN_COV  = "data/pantheon_plus/SN_MATCHED_sigma2.diag.csv"

# ---- runner ----
PYBIN  = os.environ.get("PYTHON_BIN", sys.executable)  # your venv python if exported
RUNNER = "tools/run_joint_resonant_lock_fixed.py"

# Base command; placeholders: %OUT%  %F%  %ACLAMP%
BASE_CMD = (
    "PYTHONPATH=. %PY% -u {runner} "
    f"--bao-csv {BAO_CSV} --bao-cov {BAO_COV} "
    f"--sn-csv {SN_CSV} --sn-cov {SN_COV} "
    "--bao-only --optimizer lbfgsb --max-evals 500 "
    "--prior-f-mean %F% --prior-f-sigma 0.03 %ACLAMP% "
    "--out-prefix %OUT%"
)

WROTE_RE = re.compile(r"Wrote\s+([^\s]+\.json)")
CHI2_95 = 3.84

# ---------- helpers ----------
def _load_json_strict(out_prefix: str, f_val: float, cmd: str, timeout: float) -> dict:
    """Run the command; only accept the expected file or the path echoed as 'Wrote …'."""
    proc = sp.run(["bash","-lc",cmd], stdout=sp.PIPE, stderr=sp.PIPE, timeout=timeout)
    exp = pathlib.Path(out_prefix + ".json")
    if exp.exists():
        return json.load(exp.open())
    m = WROTE_RE.search(proc.stdout.decode("utf-8","ignore"))
    if m:
        alt = pathlib.Path(m.group(1))
        if alt.exists():
            return json.load(alt.open())
    raise RuntimeError(
        f"No JSON for f={f_val}. "
        f"stderr[:160]={proc.stderr.decode('utf-8','ignore')[:160]!r} "
        f"stdout[:160]={proc.stdout.decode('utf-8','ignore')[:160]!r}"
    )

def _chi2_bao(J: dict) -> float:
    return float(J.get("breakdown", {}).get("chi2_bao", J.get("chi2")))

def _chi2_total(J: dict) -> float:
    return float(J["chi2"])

def _A_best(J: dict) -> float:
    return float(J.get("best_params", {}).get("A", float("nan")))

def _fmt(x, digits=6):
    try:
        if x is None: return "nan"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return "nan"
        return f"{x:.{digits}g}"
    except Exception:
        return "nan"

def _cmd_for(out_prefix: str, f_val: float, clamp: str) -> str:
    return (BASE_CMD
            .replace("%PY%", PYBIN)
            .replace("{runner}", RUNNER)
            .replace("%OUT%", out_prefix)
            .replace("%F%", f"{f_val:g}")
            .replace("%ACLAMP%", clamp)
            .strip())

def run_free(f_val: float, outdir: pathlib.Path, timeout: float) -> dict:
    outp = str(outdir / f"f{f_val:.4f}_free")
    return _load_json_strict(outp, f_val, _cmd_for(outp, f_val, clamp=""), timeout)

def run_fixed(f_val: float, A: float, outdir: pathlib.Path, timeout: float) -> dict:
    outp = str(outdir / f"f{f_val:.4f}_A{A:.6g}")
    clamp = f"--A0 {A:g} --prior-A-sigma 1e-12"
    return _load_json_strict(outp, f_val, _cmd_for(outp, f_val, clamp=clamp), timeout)

def analytic_limit(A_best: float, chi2_best_bao: float, chi2_null_bao: float) -> float:
    d = chi2_null_bao - chi2_best_bao
    if not (math.isfinite(d) and d > 0 and math.isfinite(A_best)):
        return float("nan")
    return A_best + math.sqrt(CHI2_95) * (abs(A_best) / math.sqrt(d))

def profile_limit_total(f_val: float, A_best: float, chi2_best_total: float,
                        outdir: pathlib.Path, timeout: float) -> float:
    """Exact one-sided 95% using TOTAL χ² crossing (matches analytic scale)."""
    if not math.isfinite(A_best):
        return float("nan")

    def chi2_tot_at(A: float) -> float:
        return _chi2_total(run_fixed(f_val, A, outdir, timeout))

    # seed with analytic step using TOTAL gap (free vs A=0)
    dnull = chi2_tot_at(0.0) - chi2_best_total
    if not (math.isfinite(dnull) and dnull > 0.0):
        return float("nan")
    sigma_A = abs(A_best) / math.sqrt(dnull)
    A_seed = max(A_best + math.sqrt(CHI2_95) * sigma_A, A_best * 1.05)

    # bracket upwards
    A_lo = max(0.0, A_best)
    A_hi = min(1.0, A_seed)
    chi_hi = chi2_tot_at(A_hi)
    tries = 0
    while (chi_hi - chi2_best_total) < CHI2_95 and A_hi < 1.0 and tries < 20:
        A_lo = A_hi
        A_hi = min(1.0, max(A_hi * 1.7, A_hi + 1e-4))
        chi_hi = chi2_tot_at(A_hi)
        tries += 1
    if (chi_hi - chi2_best_total) < CHI2_95:
        return float("nan")

    # bisection
    for _ in range(45):
        A_mid = 0.5 * (A_lo + A_hi)
        d = chi2_tot_at(A_mid) - chi2_best_total
        if abs(d - CHI2_95) < 5e-3 or abs(A_hi - A_lo) < 5e-5:
            return A_mid
        if d < CHI2_95:
            A_lo = A_mid
        else:
            A_hi = A_mid
    return 0.5 * (A_lo + A_hi)

# ---- main ----
def main():
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(description="BAO amplitude upper limits vs f (analytic + profile).")
    ap.add_argument("--f-min", type=float, default=2.00)
    ap.add_argument("--f-max", type=float, default=3.50)
    ap.add_argument("--f-step", type=float, default=0.25)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--outdir", type=str, default="outputs/bao_limits_profile")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    F = np.round(np.arange(args.f_min, args.f_max + 1e-12, args.f_step), 10)

    rows = []
    for f_val in F:
        try:
            Jf   = run_free(f_val, outdir, args.timeout)
            Ab   = _A_best(Jf)
            cB   = _chi2_bao(Jf)
            cTot = _chi2_total(Jf)

            cNullB = _chi2_bao(run_fixed(f_val, 0.0, outdir, args.timeout))
            dB     = cNullB - cB

            A95a = analytic_limit(Ab, cB, cNullB)                         # BAO-only analytic
            A95p = profile_limit_total(f_val, Ab, cTot, outdir, args.timeout)  # TOTAL χ² profile

            rows.append({
                "f": f_val,
                "A_best": Ab,
                "chi2_best_bao": cB,
                "chi2_null_bao": cNullB,
                "delta_chi2": dB,
                "A95_up_analytic": A95a,
                "A95_up_profile": A95p,
            })

            print(f"f={f_val:.2f}  A_best≈{_fmt(Ab)}  Δχ²(BAO)={dB:.3f}  "
                  f"A95↑(analytic)≈{_fmt(A95a)}  A95↑(profile)≈{_fmt(A95p)}")

        except Exception as e:
            print(f"f={f_val:.2f}  error: {e}")

    out_csv = outdir / "bao_limits_profile.csv"
    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "f","A_best","chi2_best_bao","chi2_null_bao","delta_chi2","A95_up_analytic","A95_up_profile"
        ])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote: {out_csv}")

    if args.plot and rows:
        try:
            import pandas as pd, matplotlib.pyplot as plt
            df = pd.DataFrame(rows).sort_values("f")

            # Plot A_best vs f
            fig1 = df.plot(
                x="f", y="A_best", marker="o", legend=False, title="Best-fit A vs f"
            ).get_figure()
            fig1.savefig(outdir / "A_best_vs_f.png", dpi=160)

            # Plot both 95% limits
            ax = df.rename(columns={
                "A95_up_analytic": "analytic",
                "A95_up_profile": "profile"
            }).plot(
                x="f", y=["analytic", "profile"], marker="o", title="95% upper limit on A vs f"
            )
            ax.get_figure().savefig(outdir / "A95_vs_f.png", dpi=160)

            print(f"Wrote: {outdir/'A_best_vs_f.png'} and {outdir/'A95_vs_f.png'}")

        except Exception as e:
            print(f"Plotting skipped (error): {e}")

if __name__ == "__main__":
    main()
