import numpy as np

def gamma_smooth(z, gamma0=1.50, g1=0.0):
    """Bounded gamma evolution: γ(z) = γ0 + g1 * z/(1+z)."""
    z = np.asarray(z, dtype=float)
    return gamma0 + g1 * (z / (1.0 + z))

PROFILES = {
    "smooth": gamma_smooth,
}
