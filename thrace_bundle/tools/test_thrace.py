
# ============================
# File: tools/test_thrace.py
# ----------------------------
"""Minimal tests for Thrace (no external files required). Run:

PYTHONPATH=. python3 -m tools.test_thrace

"""
import math
import unittest
import pandas as pd

from tools.thrace_recycling_model import RecyclingModel
from tools.thrace_fit import fit_recycling_model


class TestThrace(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"z": [0.0, 0.5, 1.0, 2.0], "rho_bh": [1e5, 2e5, 3e5, 5e5]})

    def test_model_runs(self):
        m = RecyclingModel()
        sol = m.solve_from_rho_bh(self.df)
        self.assertTrue({"z", "rho_de", "rho_dm"}.issubset(set(sol.columns)))
        self.assertGreater(len(sol), 3)
        self.assertTrue(sol[["rho_de", "rho_dm"]].to_numpy().min() >= 0.0)

    def test_fit_succeeds_without_rho_tot(self):
        m = RecyclingModel()
        sol = m.solve_from_rho_bh(self.df)
        res = fit_recycling_model(self.df, sol, {"alpha": m.alpha, "gamma": m.gamma, "beta_de": m.beta_de, "beta_dm": m.beta_dm})
        self.assertIn("chi2", res)
        self.assertIn("verdict", res)

    def test_guard_against_bad_z(self):
        bad = pd.DataFrame({"z": [-0.9999999, 0.0], "rho_bh": [1.0, 1.0]})
        m = RecyclingModel()
        sol = m.solve_from_rho_bh(bad)
        self.assertGreater(len(sol), 0)


if __name__ == "__main__":
    unittest.main()
