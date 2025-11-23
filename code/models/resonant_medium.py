"""
resonant_medium.py — toy background+growth for 'medium + resonance' cosmology

We parameterize an effective dark sector with two pieces:
  (i) a cold medium (behaves like CDM at background level);
  (ii) a long-wavelength resonant mode with effective equation of state w_res(a)
       that can induce mild oscillatory features in H(a) and modify linear growth.

This module provides:
  - H(a, params): background expansion
  - growth_fsigma8(z, params): linear growth via differential equation for D(a)
  - loglike_* functions for SN, BAO, growth (placeholders)

NOTE: This is a scaffold to be wired to real likelihoods and data loaders.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class Params:
    H0: float = 67.4
    Omega_m0: float = 0.315
    Omega_b0: float = 0.049
    sigma8_0: float = 0.81
    # Resonant sector parameters
    A_res: float = 0.0    # amplitude of resonance in fractional energy density
    f_res: float = 0.0    # log-frequency of oscillation in ln a
    phi_res: float = 0.0  # phase
    w_floor: float = -1.0 # baseline EoS of the resonant component

def E2_of_a(a, p: Params):
    """Dimensionless H(a)^2/H0^2 with a phenomenological resonant DE term.
    ρ_res(a) ~ ρ_res0 * a^{-3(1+w_eff(a))} with w_eff(a) = w_floor + A_res * sin(f_res * ln a + phi_res).
    """
    Om = p.Omega_m0
    Ol = 1.0 - Om
    # effective w(a)
    w_eff = p.w_floor + p.A_res * np.sin(p.f_res * np.log(a + 1e-30) + p.phi_res)
    # integrate d ln ρ_res / d ln a = -3(1+w_eff)
    # approximate with average over small step (use continuous formula for simplicity)
    # Treat resonant sector as fraction of Ol today
    Ol0 = Ol
    # Compute scaling factor for resonant density (approximate integral):
    # ρ_res(a) = ρ_res0 * a^{-3(1+w_floor)} * exp[ 3 A_res/f_res * (cos(f_res ln a + phi_res) - cos(phi_res)) ]
    eps = 1e-9 if abs(p.f_res) < 1e-6 else p.f_res
    osc_term = np.exp(3.0 * p.A_res/eps * (np.cos(eps*np.log(a+1e-30) + p.phi_res) - np.cos(p.phi_res)))
    rho_res = Ol0 * a**(-3*(1.0+p.w_floor)) * osc_term
    return Om * a**(-3) + rho_res

def H_of_z(z, p: Params):
    a = 1.0/(1.0+z)
    return p.H0 * np.sqrt(E2_of_a(a, p))

def growth_fsigma8(z, p: Params):
    """Solve the linear growth equation for D(a) in GR on this background.
    d^2 D/d a^2 + [ (3/a) + (d ln H/d ln a)/a ] dD/da - 1.5 * Ω_m(a)/a^2 * D = 0
    Then return fσ8(z) = f(z) * σ8(z).
    """
    a_grid = np.linspace(1e-4, 1.0, 4000)
    # Compute H(a) and Ω_m(a)
    E2 = E2_of_a(a_grid, p)
    H = np.sqrt(E2)
    Om_a = (p.Omega_m0 * a_grid**-3) / E2
    # d ln H / d ln a
    dlnH_dlna = np.gradient(np.log(H+1e-30), np.log(a_grid+1e-30))
    # Integrate D(a) with simple finite-difference (shooting with D ~ a at early times)
    D = np.zeros_like(a_grid); dD = np.zeros_like(a_grid)
    D[0] = a_grid[0]; dD[0] = 1.0
    for i in range(1, len(a_grid)):
        a = a_grid[i]
        da = a_grid[i]-a_grid[i-1]
        A = (3.0/a + dlnH_dlna[i]/a)
        B = -1.5*Om_a[i]/(a*a)
        # simple Euler step for illustrative purposes
        d2D = -A*dD[i-1] - B*D[i-1]
        dD[i] = dD[i-1] + d2D*da
        D[i] = D[i-1] + dD[i-1]*da
    # normalize D(a=1)=1
    D /= (D[-1] + 1e-30)
    # σ8(z) ~ σ8_0 * D(a)
    def interp(x,y,xi):
        return np.interp(xi, x, y)
    def f_of_a(a):
        # f = d ln D / d ln a
        dlnD_dlna = np.gradient(np.log(D+1e-30), np.log(a_grid+1e-30))
        return np.interp(a, a_grid, dlnD_dlna)
    a = 1.0/(1.0+z)
    f = f_of_a(a)
    sig8 = p.sigma8_0 * interp(a_grid, D, a)
    return f*sig8
