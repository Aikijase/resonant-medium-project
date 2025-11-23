
# ============================
# File: tools/thrace_recycling_model.py
# ----------------------------
"""
Simple ODE-based RecyclingModel.
Model variables: rho_de(z), rho_dm(z) sourced by rho_bh(z) via kernel/filter.
This file intentionally keeps things explicit and small so you can extend.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import odeint


class RecyclingModel:
    """A toy ODE-based recycling model.

    d rho_de / d t = + alpha * rho_bh(t) - beta_de * rho_de
    d rho_dm / d t = + gamma * rho_bh(t) - beta_dm * rho_dm

    We treat 't' as a simple monotonic proxy of redshift (t~1/(1+z)).
    Replace with proper cosmological t(z) for production.
    """

    def __init__(self, alpha: float = 1e-3, gamma: float = 1e-3, beta_de: float = 1e-2, beta_dm: float = 1e-3):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.beta_de = float(beta_de)
        self.beta_dm = float(beta_dm)

    @staticmethod
    def z_to_time(z):
        z = np.asarray(z, dtype=float)
        # avoid division by zero if z == -1 (shouldn't occur in cosmology, but guard anyway)
        return 1.0 / (1.0 + np.clip(z, -0.999999, None))

    def solve_from_rho_bh(self, df_rho_bh: pd.DataFrame, z_eval=None) -> pd.DataFrame:
        """Given dataframe with columns ['z','rho_bh'], return dataframe with z,rho_de,rho_dm"""
        if "z" not in df_rho_bh.columns or "rho_bh" not in df_rho_bh.columns:
            raise ValueError("expect columns z and rho_bh in input df")

        df = df_rho_bh.sort_values("z", ascending=False).reset_index(drop=True)  # high z -> low
        zs = df["z"].values
        rhos = df["rho_bh"].values

        # map z to time
        ts = self.z_to_time(zs)
        t_min, t_max = ts.min(), ts.max()
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min == t_max:
            raise ValueError("Degenerate time grid from z->t mapping. Check input z range.")

        # interpolation function for rho_bh(t)
        rho_interp = interp1d(ts, rhos, kind="linear", fill_value="extrapolate", assume_sorted=True)

        def rhs(y, t):
            rho_de, rho_dm = y
            rho_bh_t = float(rho_interp(t))
            drho_de_dt = self.alpha * rho_bh_t - self.beta_de * rho_de
            drho_dm_dt = self.gamma * rho_bh_t - self.beta_dm * rho_dm
            return [drho_de_dt, drho_dm_dt]

        # integrate on a regular time grid (from high-z to low-z in our simple mapping)
        n_grid = max(200, len(ts))
        t_grid = np.linspace(t_min, t_max, n_grid)
        y0 = [0.0, 0.0]
        sol = odeint(rhs, y0, t_grid)

        # map back to redshift via inverse of z_to_time (simple)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_grid = 1.0 / t_grid - 1.0
        # guard any infs from numerical edges
        mask = np.isfinite(z_grid)
        z_grid = z_grid[mask]
        sol = sol[mask]

        out = pd.DataFrame({"z": z_grid, "rho_de": sol[:, 0], "rho_dm": sol[:, 1]})
        out = out.sort_values("z", ascending=True).reset_index(drop=True)
        return out


if __name__ == "__main__":
    # quick local smoke test
    import pandas as pd
    df = pd.DataFrame({"z": [0.0, 0.5, 1.0, 2.0], "rho_bh": [1e5, 2e5, 3e5, 5e5]})
    model = RecyclingModel()
    sol = model.solve_from_rho_bh(df)
    print(sol.head())
