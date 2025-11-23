import numpy as np
from scipy.integrate import solve_ivp

EPS = 1e-9

def H_of_z(z, p):
    """
    Placeholder resonant H(z). Must stay positive; we clamp to avoid log/ODE blowups.
    Replace with your calibrated mapping when ready.
    """
    A, f, phi, gamma = p["A"], p["f"], p["phi"], p["gamma"]
    H = np.exp(-gamma*z) * (1.0 + A*np.cos(2*np.pi*f*z + phi))
    return np.clip(H, EPS, None)

def dlnH_dlnA(z, p, dz=1e-4):
    """
    Compute d(ln H)/d(ln a) via chain rule:
      d ln H / d ln a = -(1+z) * d ln H / dz
    Use a symmetric difference in z; guard H > 0.
    """
    z1, z2 = z + dz, z - dz
    H1 = H_of_z(z1, p)
    H2 = H_of_z(z2, p)
    dlnH_dz = (np.log(H1) - np.log(H2)) / (2.0*dz)
    return -(1.0 + z) * dlnH_dz

def solve_growth(p, zmax=2.0, npts=200, sigma8_0=0.8):
    """
    Solve linear growth D(a) with GR-like clustering (mu=1), driven by H(z).
    Returns dict with z (ascending), D (normalized so D[z=0]=1), and f*sigma8.
    """
    # Build an ASCENDING grid in ln a (important for gradient & ODE stability)
    a_min, a_max = 1.0/(1.0 + zmax), 1.0
    ln_a_grid = np.linspace(np.log(a_min), np.log(a_max), npts)  # increasing
    a_grid = np.exp(ln_a_grid)
    z_grid = 1.0/a_grid - 1.0  # ascending  z:  z(a_min) -> z=0

    # ODE in ln a:
    #   D'' + (2 + dlnH/dlnA) D' - (3/2) D = 0   (schematic GR form)
    # where ' is derivative wrt ln a.
    def ode(ln_a, y):
        a = np.exp(ln_a)
        z = 1.0/a - 1.0
        D, Dp = y
        dlnH = dlnH_dlnA(z, p)
        Dpp = - (2.0 + dlnH) * Dp + 1.5 * D
        return [Dp, Dpp]

    sol = solve_ivp(
        ode,
        (ln_a_grid[0], ln_a_grid[-1]),
        y0=[1.0, 0.0],               # normalize later anyway
        t_eval=ln_a_grid,
        rtol=1e-6, atol=1e-8,
        method="RK45"
    )
    D = sol.y[0]
    # Normalize so D(z=0) = 1
    D /= max(D[-1], EPS)

    # f = d ln D / d ln a  (use ascending ln a grid)
    f_log = np.gradient(np.log(np.clip(D, EPS, None)), ln_a_grid, edge_order=2)
    fs8 = sigma8_0 * f_log * D

    return {"z": z_grid, "D": D, "fs8": fs8}
