# utils_node_models.py
import numpy as np

def resonant_modifier(k, tau, omega, phase=0.0):
    """Single-phase modifier: exp(-tau k^2) * cos^2(omega k + phase)."""
    return np.exp(-tau * k**2) * (np.cos(omega * k + phase)**2)

def resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=np.pi/10):
    """
    Average cos^2 over a narrow phase window around 0 to smooth wiggles.
    width ~ pi/16 by default. n_phase odd to include 0.
    """
    phases = np.linspace(-width, width, n_phase)
    acc = 0.0
    base = np.exp(-tau * k**2)
    for ph in phases:
        acc += (np.cos(omega * k + ph)**2)
    return base * (acc / len(phases))

def place_node_at(k_star):
    """Return omega that places the first node near k_star: (pi/2)/omega = k_star -> omega = pi/(2 k_star)."""
    return np.pi/(2.0*k_star)
