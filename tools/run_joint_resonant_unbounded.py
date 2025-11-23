#!/usr/bin/env python3
"""
run_joint_resonant_unbounded.py

Thin wrapper around tools/run_joint_resonant_lock_fixed.py that:
  • widens the gamma (γ) bound to (0, 3) instead of (0, 1)
  • adds a true hard-fix option: --fix-gamma-to <value>
  • guarantees bounds length always matches the optimizer x0 length
    (so no more "bounds/x0 length" broadcast errors)
  • otherwise defers all model math, I/O, and JSON writing to the base runner

Usage is identical to run_joint_resonant_lock_fixed.py, plus:
  --fix-gamma-to <float>  # pins γ to this exact value

Examples:
  PYTHONPATH=. ./.venv/bin/python -u tools/run_joint_resonant_unbounded.py \
    --bao-csv data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv \
    --bao-cov data/desi_dr1_bao/bao_covariance_plus_lya.csv \
    --sn-csv  data/pantheon_plus/sn_MATCHED_mu.csv \
    --sn-cov  data/pantheon_plus/SN_MATCHED_sigma2.diag.csv \
    --optimizer lbfgsb --max-evals 1000 \
    --prior-f-mean 2.6 --prior-f-sigma 0.05 \
    --out-prefix outputs/joint_gamma_free_unbounded

  # Fix γ to 2.3 (true hard constraint)
  PYTHONPATH=. ./.venv/bin/python -u tools/run_joint_resonant_unbounded.py \
    ...same data/opts... \
    --fix-gamma-to 2.3 \
    --out-prefix outputs/joint_gamma_fix_2p3
"""

from __future__ import annotations
import sys, math
from typing import Optional

# Import the existing runner as "base"
from tools import run_joint_resonant_lock_fixed as base

# --- 1) Parse our extra flag early and strip it from sys.argv so base's argparse ignores it
FIX_GAMMA_TO: Optional[float] = None
argv = sys.argv[:]
if "--fix-gamma-to" in argv:
    i = argv.index("--fix-gamma-to")
    try:
        FIX_GAMMA_TO = float(argv[i+1])
    except Exception:
        print("ERROR: --fix-gamma-to requires a numeric value", file=sys.stderr)
        sys.exit(2)
    # Remove the flag and its value so base's argparse doesn't see it
    del argv[i:i+2]
    sys.argv = argv

# --- 2) Patch base.minimize with widened γ bound + robust bounds construction
def _patched_minimize(chi2, x0, method, maxevals, fit_scales, shared_scale, fix_gamma):
    """
    Mirrors base.minimize signature but:
      - widens γ upper bound to 3.0
      - if fix_gamma is True, sets γ to 0 exactly
      - if FIX_GAMMA_TO is not None, sets γ to that exact value
      - ensures bounds list length == len(x0) by appending scale bounds as needed
    """
    import scipy.optimize as opt

    n = len(x0)
    # Order assumed by the base runner: [A, f, phi, gamma, (scale1), (scale2)]
    bounds = [
        (0.0, 2.0),              # A
        (1.0, 6.0),              # f
        (-math.pi, math.pi),     # phi
        (0.0, 3.0),              # gamma (widened)
    ]

    # Add scale bounds if x0 includes them (shared or two independent)
    if n >= 5:
        bounds.append((0.8, 1.2))    # s_shared or s_DM
    if n >= 6:
        bounds.append((0.8, 1.2))    # s_DH

    # Hard constraints on γ
    if fix_gamma:
        bounds[3] = (0.0, 0.0)
    if FIX_GAMMA_TO is not None:
        bounds[3] = (FIX_GAMMA_TO, FIX_GAMMA_TO)

    # Final guard: match exactly the optimizer dimensionality
    bounds = bounds[:n]

    return opt.minimize(
        chi2, x0, method="L-BFGS-B",
        bounds=bounds,
        options={"maxfun": maxevals}
    )

# Monkey-patch the base runner's minimize
base.minimize = _patched_minimize

# --- 3) Run the original main() with our patch in place
if __name__ == "__main__":
    base.main()
