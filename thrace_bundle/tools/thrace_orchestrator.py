
# ============================
# File: tools/thrace_orchestrator.py
# ----------------------------
"""
Orchestrator for Thrace phase: runs recycling ODE, creates fit.json, and assembles outputs.
Drop into your project `tools/` folder and run from project root with `PYTHONPATH=.`

Outputs (created under outputs/thrace/):
 - outputs/thrace/recycling.solution.csv
 - outputs/thrace/recycling.fit.json
 - outputs/thrace/recycling.report.txt

Usage examples:
  PYTHONPATH=. python3 tools/thrace_orchestrator.py --config configs/thrace_default.yaml
  PYTHONPATH=. python3 tools/thrace_orchestrator.py --rho-bh data/rho_bh_z.clean.csv

Design principles:
 - Minimal dependencies (numpy, scipy, pandas, pyyaml). Keep scripts drop-in.
 - Clear JSON metrics for handover and CI.
 - Import-path robust: works whether you call from project root or tools/ directly.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# --- Robust local imports ---
# Support both "python tools/thrace_orchestrator.py" and "PYTHONPATH=. python tools/thrace_orchestrator.py"
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:  # package-style
    from tools.thrace_recycling_model import RecyclingModel
    from tools.thrace_fit import fit_recycling_model
except Exception:  # flat-style fallback
    from thrace_recycling_model import RecyclingModel
    from thrace_fit import fit_recycling_model


def ensure_dirs():
    (_PROJECT_ROOT / "outputs/thrace").mkdir(parents=True, exist_ok=True)


def load_rho_bh(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    # expect columns: z, rho_bh  (rho_tot optional)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML with rho_bh_path and optional parameters")
    p.add_argument("--rho-bh", dest="rho_bh", default=None, help="CSV with columns z,rho_bh[,rho_tot]")
    p.add_argument("--out-prefix", default="outputs/thrace/recycling")
    # allow quick param overrides without editing YAML
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--beta-de", dest="beta_de", type=float, default=None)
    p.add_argument("--beta-dm", dest="beta_dm", type=float, default=None)
    args = p.parse_args()

    ensure_dirs()

    cfg = {}
    if args.config:
        try:
            import yaml
        except ImportError as e:
            print("ERROR: pyyaml is required for --config. Try: pip install pyyaml", file=sys.stderr)
            raise
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    rho_path = args.rho_bh or cfg.get("rho_bh_path")
    if rho_path is None:
        print("ERROR: provide --rho-bh or a config with rho_bh_path", file=sys.stderr)
        sys.exit(2)

    df_rho = load_rho_bh(rho_path)

    # model params: CLI overrides > config > defaults
    mparams = {
        "alpha": args.alpha if args.alpha is not None else cfg.get("alpha", 1e-3),
        "gamma": args.gamma if args.gamma is not None else cfg.get("gamma", 1e-3),
        "beta_de": args.beta_de if args.beta_de is not None else cfg.get("beta_de", 1e-2),
        "beta_dm": args.beta_dm if args.beta_dm is not None else cfg.get("beta_dm", 1e-3),
    }

    model = RecyclingModel(**mparams)
    sol_df = model.solve_from_rho_bh(df_rho)

    out_prefix = args.out_prefix
    out_csv = f"{out_prefix}.solution.csv"
    sol_df.to_csv(out_csv, index=False)

    # run a fit
    fit_results = fit_recycling_model(df_rho, sol_df, mparams)

    out_json = f"{out_prefix}.fit.json"
    with open(out_json, "w") as f:
        json.dump(fit_results, f, indent=2)

    out_report = f"{out_prefix}.report.txt"
    with open(out_report, "w") as f:
        f.write("Thrace recycling run report\n\n")
        f.write(json.dumps(fit_results, indent=2))
        f.write("\n")

    print("Wrote:")
    print(" -", out_csv)
    print(" -", out_json)
    print(" -", out_report)


if __name__ == "__main__":
    main()
