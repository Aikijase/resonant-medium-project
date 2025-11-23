#!/usr/bin/env python3
"""
End-to-end Thrace chain:

1) Run recycling orchestrator (produces rho_de/rho_dm from rho_bh)
2) Run BAO+SN fitter via tools/thrace_pipeline_fit.py (Option C)
3) Render a Markdown report
4) Make a metrics plot (chi2/AIC/BIC for LCDM vs RESN)

Artifacts are kept under: outputs/thrace_runs/<timestamp>/

Usage examples (from repo root):

  # Minimal: uses your existing joint_baseline.json and your BH CSV
  PYTHONPATH=. python tools/thrace_chain.py \
    --rho-bh data/rho_bh_z.clean.csv \
    --rho-bh-col rho_bh_Msun_per_Mpc3 \
    --config configs/joint_baseline.json \
    --tag thrace

  # With parameter overrides
  PYTHONPATH=. python tools/thrace_chain.py \
    --rho-bh data/rho_bh_z.clean.csv \
    --rho-bh-col rho_bh_Msun_per_Mpc3 \
    --alpha 0.001 --gamma 0.001 --beta-de 0.01 --beta-dm 0.001 \
    --config configs/joint_baseline.json \
    --tag thrace
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # current venv python

def _run(cmd, cwd=None, check=True):
    print("[thrace-chain] RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout, end="")
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="")
    if check and proc.returncode != 0:
        raise SystemExit(f"[thrace-chain] step failed with code {proc.returncode}")
    return proc

def main():
    ap = argparse.ArgumentParser(description="Run Thrace end-to-end and collect outputs per run.")
    # Recycling inputs
    ap.add_argument("--rho-bh", required=True, help="CSV path for BH density vs z")
    ap.add_argument("--z-col", default=None, help="Name of redshift column (default: auto)")
    ap.add_argument("--rho-bh-col", default=None, help="Name of BH density column (default: auto)")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--beta-de", dest="beta_de", type=float, default=None)
    ap.add_argument("--beta-dm", dest="beta_dm", type=float, default=None)
    # Pipeline config
    ap.add_argument("--config", required=True, help="JSON config for make_fit_json.py (positional arg)")
    ap.add_argument("--tag", default="thrace", help="Tag for the fitter/logs (default: thrace)")
    # Output base folder
    ap.add_argument("--out-base", default="outputs/thrace_runs", help="Base directory for timestamped runs")
    args = ap.parse_args()

    # Prepare run directory
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (PROJECT_ROOT / args.out_base / ts)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Recycling orchestrator -> keep inside run_dir
    orch_out_prefix = str(run_dir / "recycling")
    orch_cmd = [
        PY, str(PROJECT_ROOT / "tools" / "thrace_orchestrator.py"),
        "--rho-bh", args.rho_bh,
        "--out-prefix", orch_out_prefix,
    ]
    if args.z_col:         orch_cmd += ["--z-col", args.z_col]
    if args.rho_bh_col:    orch_cmd += ["--rho-bh-col", args.rho_bh_col]
    if args.alpha is not None:   orch_cmd += ["--alpha", str(args.alpha)]
    if args.gamma is not None:   orch_cmd += ["--gamma", str(args.gamma)]
    if args.beta_de is not None: orch_cmd += ["--beta-de", str(args.beta_de)]
    if args.beta_dm is not None: orch_cmd += ["--beta-dm", str(args.beta_dm)]

    _run(orch_cmd)

    # 2) Pipeline fitter wrapper -> write std summary directly in run_dir
    std_summary_path = run_dir / "joint.std_summary.json"
    fit_cmd = [
        PY, str(PROJECT_ROOT / "tools" / "thrace_pipeline_fit.py"),
        "--config", args.config,
        "--tag", args.tag,
        "--std-out", str(std_summary_path),
        # Let the wrapper search normally under outputs/
        "--search-dir", "outputs",
    ]
    _run(fit_cmd)

    # 3) Report -> Markdown in run_dir
    md_path = run_dir / "joint.report.md"
    rep_cmd = [
        PY, str(PROJECT_ROOT / "tools" / "thrace_report.py"),
        "--summary", str(std_summary_path),
        "--md", str(md_path),
    ]
    _run(rep_cmd)

    # 4) Plot -> saved next to summary by the plotter (we’ll keep in run_dir)
    # The plotter saves to summary.parent; so just call it with our std_summary in run_dir
    plot_cmd = [
        PY, str(PROJECT_ROOT / "tools" / "thrace_plot.py"),
        "--summary", str(std_summary_path),
    ]
    _run(plot_cmd)

    # 5) Copy the raw fit.json into the run folder for completeness (if found)
    # Our fitter usually writes outputs/<something>.fit.json; locate newest and copy.
    fit_json_guess = None
    outputs_dir = PROJECT_ROOT / "outputs"
    fits = sorted(outputs_dir.rglob("*.fit.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if fits:
        fit_json_guess = fits[0]
        try:
            shutil.copy2(fit_json_guess, run_dir / fit_json_guess.name)
        except Exception:
            pass

    # 6) Emit a tiny index.json with pointers
    index = {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "std_summary": str(std_summary_path.relative_to(PROJECT_ROOT)),
        "report_md": str(md_path.relative_to(PROJECT_ROOT)),
        "recycling_csv": f"{orch_out_prefix}.solution.csv",
        "fit_json_guess": str(fit_json_guess.relative_to(PROJECT_ROOT)) if fit_json_guess else None,
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2))

    print("\n[thrace-chain] DONE")
    print(" run_dir          :", index["run_dir"])
    print(" std_summary.json :", index["std_summary"])
    print(" report.md        :", index["report_md"])
    print(" recycling CSV    :", index["recycling_csv"])
    if fit_json_guess:
        print(" fit.json         :", index["fit_json_guess"])

if __name__ == "__main__":
    main()
