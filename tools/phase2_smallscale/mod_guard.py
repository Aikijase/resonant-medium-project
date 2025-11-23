import numpy as np
import os, sys
# Ensure this file's directory is on sys.path so sibling modules import cleanly
sys.path.append(os.path.dirname(__file__))

from utils_node_models import resonant_modifier_phase_smeared

def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3 - 2*x)

def resonant_modifier_guarded(k, tau, omega, *,
                              n_phase=9, width=np.pi/8,
                              k_guard=8.0, delta=0.7):
    """
    Low-k guard for Ly-α safety:
      - k <= k_guard:        mod = 1
      - k >= k_guard+delta:  mod = resonant modifier
      - smooth blend in between (C1 smoothstep)
    """
    k = np.asarray(k, float)
    mod_raw = resonant_modifier_phase_smeared(k, tau, omega, n_phase=n_phase, width=width)
    # Blend from 0 (below guard) to 1 (above guard+delta)
    t = (k - k_guard) / max(delta, 1e-6)
    w = _smoothstep(t)
    return (1.0 - w) + w * mod_raw
