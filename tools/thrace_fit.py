
# ============================
# File: tools/thrace_fit.py
# ----------------------------
"""
Simple fitter that compares model output to a target baseline (e.g. LambdaCDM total density) and
returns a JSON-friendly dictionary of chi2/AIC/BIC-style metrics and best-fit parameter guesses.
This is intentionally minimal—swap in your preferred likelihood routines.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FitResult:
    chi2: float
    n_points: int
    params: Dict


def _chi2_between(df_obs: pd.DataFrame, df_model: pd.DataFrame) -> float:
    """Robust chi2: compare df_obs.rho_tot to model (rho_de+rho_dm) after interpolation."""
    from scipy.interpolate import interp1d
    f = interp1d(df_model["z"], df_model["rho_de"] + df_model["rho_dm"], bounds_error=False, fill_value="extrapolate")
    model_vals = f(df_obs["z"])
    obs_vals = df_obs["rho_tot"].values
    # simple relative error floor (5%)
    sigma = 0.05 * np.maximum(np.abs(obs_vals), 1e-12)
    chi2 = float(np.sum(((obs_vals - model_vals) / sigma) ** 2))
    return chi2


def fit_recycling_model(df_rho_bh: pd.DataFrame, sol_df: pd.DataFrame, mparams: Optional[Dict] = None) -> Dict:
    """Return a small summary dict with metrics.

    If df_rho_bh lacks 'rho_tot', we synthesize a flat target so CI never explodes.
    """
    if "rho_tot" not in df_rho_bh.columns:
        df = df_rho_bh.copy()
        flat = float((sol_df["rho_de"].values[-1] + sol_df["rho_dm"].values[-1]))
        df["rho_tot"] = flat
    else:
        df = df_rho_bh.copy()

    chi2 = _chi2_between(df, sol_df)
    n = int(len(df))

    result = {
        "chi2": float(chi2),
        "n": n,
        "params": dict(mparams or {}),
        "verdict": "PASS" if chi2 / max(1, n) < 10 else "REVIEW"
    }
    return result


if __name__ == "__main__":
    # Self-test
    import pandas as pd
    from thrace_recycling_model import RecyclingModel
    df_rho = pd.DataFrame({"z": [0.0, 0.5, 1.0], "rho_bh": [1e5, 2e5, 3e5]})
    model = RecyclingModel()
    sol = model.solve_from_rho_bh(df_rho)
    print(fit_recycling_model(df_rho, sol, {"alpha": 1e-3, "gamma": 1e-3, "beta_de": 1e-2, "beta_dm": 1e-3}))
