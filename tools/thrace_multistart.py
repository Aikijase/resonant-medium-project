#!/usr/bin/env python3
"""
Thrace Multistart Search (verbose, log-per-run)

- Samples (alpha, gamma, beta_de, beta_dm) unless fixed via flags
- For each sample, runs the full chain (orchestrator → fitter → summary/report/plot)
- Writes a leaderboard CSV + JSONL
- Writes a per-run log under outputs/thrace_multistart/logs/run_XXX.log
- Prints clear progress, successes, and failures

Usage (examples):

PYTHONPATH=. python tools/thrace_multistart.py \
  --rho-bh data/rho_bh_z.clean.csv \
  --rho-bh-col rho_bh_Msun_per_Mpc3 \
  --config configs/joint_baseline.json \
  --runs 12 --seed 42

# Fix some params, sample the rest:
PYTHONPATH=. python tools/thrace_multistart.py \
  --rho-bh data/rho_bh_z.clean.csv \
  --rho-bh-col rho_bh_Msun_per_Mpc3 \
  --config configs/joint_baseline.json \
  --runs 8 --alpha 0.001 --gamma 0.001
"""
from __future__ import annotations
import argparse, json, math, os, random, subprocess, sys, time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# ---------- helpers ----------

def _ensure_path(label: str, p: Path) -> None:
    if not p.exists():
        raise SystemExit(f"[multistart] ERROR: {label} not found: {p}")

def _run_and_tee(cmd: List[str], logfile: Path) -> subprocess.CompletedProcess:
    """Run a command, capture stdout/stderr, write both to logfile, and return the process."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("[multistart] RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    with logfile.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        if proc.stdout:
            f.write("\n[stdout]\n")
            f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n")
            f.write(proc.stderr)
        f.write(f"\n[exit_code] {proc.returncode}\n")
    # Echo short status to console
    if proc.returncode == 0:
        print(f"[multistart] ok → {logfile}", flush=True)
    else:
        print(f"[multistart] FAILED (code {proc.returncode}) → {logfile}", flush=True)
    return proc

def _loguniform(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("log-uniform bounds must be > 0")
    import math
    lo, hi = math.log(a), math.log(b)
    import random as _r
    return math.exp(_r.uniform(lo, hi))

def _sample_or_fix(fixed: Optional[float], rng: Tuple[float,float]) -> float:
    return float(fixed) if fixed is not None else _loguniform(*rng)

def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _extract_metrics(std_summary: Dict[str, Any]) -> Dict[str, Any]:
    models = std_summary.get("models", {}) or {}
    lcdm = models.get("lcdm", {}) or {}
    resn = models.get("resn", {}) or {}
    canon = resn if (resn.get("chi2") is not None) else lcdm
    return {
        "chi2": canon.get("chi2"),
        "AIC":  canon.get("AIC"),
        "BIC":  canon.get("BIC"),
        "dChi2": std_summary.get("dChi2"),
        "dAIC":  std_summary.get("dAIC"),
        "dBIC":  std_summary.get("dBIC"),
        "chi2_resn": resn.get("chi2"),
        "chi2_lcdm": lcdm.get("chi2"),
        "N": resn.get("N") or lcdm.get("N"),
    }

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Thrace multistart parameter search (verbose).")
    # Required I/O
    ap.add_argument("--rho-bh", required=True)
    ap.add_argument("--rho-bh-col", required=True)
    ap.add_argument("--z-col", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag-prefix", default="thrace_ms", help="prefix for fitter tag per run")
    ap.add_argument("--out-base", default="outputs/thrace_multistart", help="leaderboard + logs location")

    # Optional fixed values
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--beta-de", dest="beta_de", type=float, default=None)
    ap.add_argument("--beta-dm", dest="beta_dm", type=float, default=None)

    # Sampling ranges (log-uniform) for non-fixed params
    ap.add_argument("--alpha-range", nargs=2, type=float, default=[5e-4, 2e-3])
    ap.add_argument("--gamma-range", nargs=2, type=float, default=[5e-4, 2e-3])
    ap.add_argument("--beta-de-range", nargs=2, type=float, default=[5e-3, 5e-2])
    ap.add_argument("--beta-dm-range", nargs=2, type=float, default=[5e-4, 5e-3])

    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    rho_bh = (PROJECT_ROOT / args.rho_bh).resolve()
    cfg    = (PROJECT_ROOT / args.config).resolve()
    _ensure_path("--rho-bh", rho_bh)
    _ensure_path("--config", cfg)

    out_base = PROJECT_ROOT / args.out_base
    logs_dir = out_base / "logs"
    out_base.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_csv = out_base / "leaderboard.csv"
    leaderboard_json = out_base / "leaderboard.jsonl"

    # CSV header if new
    if not leaderboard_csv.exists():
        leaderboard_csv.write_text(
            "idx,alpha,gamma,beta_de,beta_dm,chi2,AIC,BIC,dChi2,dAIC,dBIC,chi2_resn,chi2_lcdm,N,run_dir,summary_path,log\n"
        )

    print("[multistart] ===============================================")
    print(f"[multistart] rho_bh     : {rho_bh}")
    print(f"[multistart] config     : {cfg}")
    print(f"[multistart] runs       : {args.runs} (seed={args.seed})")
    print(f"[multistart] out base   : {out_base}")
    print("[multistart] ===============================================", flush=True)

    successes, failures = 0, 0
    best: Dict[str, Any] = {}

    for i in range(1, args.runs + 1):
        # Sample (or fix) parameters
        alpha   = _sample_or_fix(args.alpha,   tuple(args.alpha_range))
        gamma   = _sample_or_fix(args.gamma,   tuple(args.gamma_range))
        beta_de = _sample_or_fix(args.beta_de, tuple(args.beta_de_range))
        beta_dm = _sample_or_fix(args.beta_dm, tuple(args.beta_dm_range))

        print(f"\n[multistart] ► Run {i}/{args.runs} "
              f"(alpha={alpha:.6g}, gamma={gamma:.6g}, beta_de={beta_de:.6g}, beta_dm={beta_dm:.6g})",
              flush=True)

        tag = f"{args.tag_prefix}_{i:03d}"
        chain_cmd = [
            PY, str(PROJECT_ROOT / "tools" / "thrace_chain.py"),
            "--rho-bh", str(rho_bh),
            "--rho-bh-col", args.rho_bh_col,
            "--config", str(cfg),
            "--tag", tag,
            "--alpha", str(alpha),
            "--gamma", str(gamma),
            "--beta-de", str(beta_de),
            "--beta-dm", str(beta_dm),
        ]
        if args.z_col:
            chain_cmd += ["--z-col", args.z_col]

        log_path = logs_dir / f"run_{i:03d}.log"
        proc = _run_and_tee(chain_cmd, log_path)

        if proc.returncode != 0:
            print(f"[multistart] run {i} failed — see {log_path}", flush=True)
            failures += 1
            continue

        # Parse run_dir from stdout (as printed by thrace_chain)
        run_dir = None
        for line in (proc.stdout or "").splitlines():
            if line.strip().startswith("run_dir") or line.strip().startswith("run_dir"):
                # accommodate both "run_dir :" with/without leading spaces
                parts = line.split(":")
                if len(parts) >= 2:
                    run_dir = parts[1].strip()
                    break
            if line.strip().startswith("run_dir"):
                run_dir = line.split(":",1)[1].strip()
                break
            if line.strip().startswith(" run_dir"):
                run_dir = line.split(":",1)[1].strip()
                break

        if not run_dir:
            # fallback: newest folder under outputs/thrace_runs
            base = PROJECT_ROOT / "outputs" / "thrace_runs"
            candidates = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            run_dir = str(candidates[0].relative_to(PROJECT_ROOT)) if candidates else None

        if not run_dir:
            print(f"[multistart] could not infer run_dir for run {i} — see {log_path}", flush=True)
            failures += 1
            continue

        run_dir_path = PROJECT_ROOT / run_dir
        summary_path = run_dir_path / "joint.std_summary.json"
        std = _read_json(summary_path)

        if not std:
            print(f"[multistart] summary missing/empty for run {i}: {summary_path} — see {log_path}", flush=True)
            failures += 1
            continue

        metrics = _extract_metrics(std)
        row = {
            "idx": i,
            "alpha": alpha, "gamma": gamma, "beta_de": beta_de, "beta_dm": beta_dm,
            "chi2": metrics.get("chi2"),
            "AIC": metrics.get("AIC"), "BIC": metrics.get("BIC"),
            "dChi2": metrics.get("dChi2"), "dAIC": metrics.get("dAIC"), "dBIC": metrics.get("dBIC"),
            "chi2_resn": metrics.get("chi2_resn"), "chi2_lcdm": metrics.get("chi2_lcdm"),
            "N": metrics.get("N"),
            "run_dir": str(run_dir_path.relative_to(PROJECT_ROOT)),
            "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
        }

        # Update best by canonical chi2 if available
        try:
            if row["chi2"] is not None and (not best or row["chi2"] < best["chi2"]):
                best = row.copy()
        except Exception:
            pass

        # Append CSV
        with leaderboard_csv.open("a") as f:
            f.write(",".join([
                str(row["idx"]),
                f"{row['alpha']:.10g}", f"{row['gamma']:.10g}",
                f"{row['beta_de']:.10g}", f"{row['beta_dm']:.10g}",
                str(row.get("chi2", "")), str(row.get("AIC", "")), str(row.get("BIC", "")),
                str(row.get("dChi2", "")), str(row.get("dAIC", "")), str(row.get("dBIC", "")),
                str(row.get("chi2_resn", "")), str(row.get("chi2_lcdm", "")),
                str(row.get("N", "")),
                row["run_dir"], row["summary_path"], row["log"],
            ]) + "\n")

        # Append JSONL
        with leaderboard_json.open("a") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[multistart] ✓ captured run {i} → χ²={row.get('chi2')}  (summary: {row['summary_path']})", flush=True)
        successes += 1

    # Final recap
    print("\n[multistart] ===============================================")
    print(f"[multistart] completed: {successes} ok, {failures} failed")
    print(f"[multistart] leaderboard CSV : {leaderboard_csv}")
    print(f"[multistart] leaderboard JSON: {leaderboard_json}")
    if best:
        print(f"[multistart] best χ² = {best['chi2']} at "
              f"(alpha={best['alpha']}, gamma={best['gamma']}, beta_de={best['beta_de']}, beta_dm={best['beta_dm']})")
        print(f"[multistart] best run_dir : {best['run_dir']}")
        print(f"[multistart] best summary: {best['summary_path']}")
        print(f"[multistart] best log    : {best['log']}")
    print("[multistart] ===============================================")

if __name__ == "__main__":
    main()
