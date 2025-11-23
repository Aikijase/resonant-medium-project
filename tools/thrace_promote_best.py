#!/usr/bin/env python3
"""
Promote the refined best (alpha, gamma) + chosen betas to a final, clean run.

Reads:
  - outputs/thrace_refine/best_row.json  (alpha, gamma)
  - uses your chosen betas (flags) or falls back to best_params.json

Writes:
  outputs/thrace_best/
    ├─ joint.std_summary.json
    ├─ joint.report.md
    ├─ joint.metrics.png
    ├─ recycling.solution.csv
    └─ index.json

Usage:
  PYTHONPATH=. python tools/thrace_promote_best.py \
    --rho-bh data/rho_bh_z.clean.csv \
    --rho-bh-col rho_bh_Msun_per_Mpc3 \
    --config configs/joint_baseline.json \
    --beta-de 0.009418881276 \
    --beta-dm 0.0008359508477 \
    --tag thrace_best
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def run(cmd):
    print("[promote] RUN:", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)

def load_best_refine():
    p = ROOT / "outputs/thrace_refine/best_row.json"
    if not p.exists():
        raise SystemExit("[promote] missing outputs/thrace_refine/best_row.json (run refine first)")
    return json.loads(p.read_text())

def load_best_ms():
    p = ROOT / "outputs/thrace_multistart/analysis/best_params.json"
    return json.loads(p.read_text()) if p.exists() else {}

def parse_run_dir(stdout: str):
    for line in (stdout or "").splitlines():
        s = line.strip()
        if s.startswith("run_dir"):
            parts = line.split(":", 1)
            if len(parts) >= 2:
                return parts[1].strip()
        if s.startswith("run_dir") or s.startswith(" run_dir"):
            parts = line.split(":", 1)
            if len(parts) >= 2:
                return parts[1].strip()
    return None

def main():
    ap = argparse.ArgumentParser(description="Promote refined best params to a clean final run.")
    ap.add_argument("--rho-bh", required=True)
    ap.add_argument("--rho-bh-col", required=True)
    ap.add_argument("--z-col", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--beta-de", type=float, default=None)
    ap.add_argument("--beta-dm", type=float, default=None)
    ap.add_argument("--tag", default="thrace_best")
    args = ap.parse_args()

    br = load_best_refine()
    ms = load_best_ms()

    alpha = float(br["alpha"])
    gamma = float(br["gamma"])
    beta_de = float(args.beta_de if args.beta_de is not None else ms.get("beta_de"))
    beta_dm = float(args.beta_dm if args.beta_dm is not None else ms.get("beta_dm"))
    if beta_de is None or beta_dm is None:
        raise SystemExit("[promote] need --beta-de/--beta-dm (or multistart best_params.json present)")

    # 1) Run full chain with fixed params
    chain = [
        PY, str(ROOT / "tools/thrace_chain.py"),
        "--rho-bh", args.rho_bh, "--rho-bh-col", args.rho_bh_col,
        "--config", args.config, "--tag", args.tag,
        "--alpha", f"{alpha}", "--gamma", f"{gamma}",
        "--beta-de", f"{beta_de}", "--beta-dm", f"{beta_dm}",
    ]
    if args.z_col:
        chain += ["--z-col", args.z_col]

    proc = run(chain)
    if proc.stdout: print(proc.stdout, end="")
    if proc.stderr: print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise SystemExit(f"[promote] chain failed with code {proc.returncode}")

    run_dir = parse_run_dir(proc.stdout)
    if not run_dir:
        raise SystemExit("[promote] could not locate run_dir from chain output")

    # 2) Copy canonical artifacts to outputs/thrace_best
    src = ROOT / run_dir
    dst = ROOT / "outputs/thrace_best"
    dst.mkdir(parents=True, exist_ok=True)

    for name in ["joint.std_summary.json", "joint.report.md", "joint.metrics.png", "recycling.solution.csv", "index.json"]:
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, dst / name)

    # 3) Write a tiny params.json for provenance
    params = {
        "alpha": alpha, "gamma": gamma,
        "beta_de": beta_de, "beta_dm": beta_dm,
        "source_run_dir": run_dir,
    }
    (dst / "params.json").write_text(json.dumps(params, indent=2))

    print("\n[promote] DONE → outputs/thrace_best/")
    for p in sorted(dst.iterdir()):
        print(" ", p.relative_to(ROOT))

if __name__ == "__main__":
    main()
