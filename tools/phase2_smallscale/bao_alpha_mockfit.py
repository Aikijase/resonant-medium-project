import numpy as np, json
from tools.phase2_smallscale.halo_mass_function import linear_pk
from tools.phase2_smallscale.utils_node_models import resonant_modifier_phase_smeared
from tools.phase2_smallscale.bao_phase_inject import apply_bao_phase_warp, dewiggle

def pk_to_xi(k, Pk, s):
    s = np.asarray(s)
    xi = np.empty_like(s, dtype=float)
    for i, si in enumerate(s):
        x = k * si
        j0 = np.sinc(x/np.pi)  # sin(x)/x
        xi[i] = np.trapz((k**2)*Pk*j0, k) / (2*np.pi**2)
    return xi

def make_mock_xi(tau=0.04, omega=0.12, eps_bao=0.006, width=np.pi/16):
    # Slightly coarser k for speed (still accurate for BAO)
    k = np.logspace(-3, 0.8, 2500)   # up to k~6.3 h/Mpc (plenty for BAO)
    P0 = linear_pk(k)
    mod= resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=width)
    P  = P0 * mod
    P  = apply_bao_phase_warp(k, P, epsilon=eps_bao)  # BAO-localized warp
    s  = np.linspace(40.0, 180.0, 701)                # BAO bump range
    xi = pk_to_xi(k, P, s)
    return k, s, xi

def xi_template_LCDM(k, s):
    P0 = linear_pk(k)
    Pnw, W = dewiggle(P0, k)
    return pk_to_xi(k, Pnw*W, s)

def fit_alpha_only_fast(s, xi_mock, k):
    """
    Cache xi_template(s) once, then for each alpha just resample at s*alpha.
    Optimize alpha by golden-section search over [0.94, 1.06].
    Broadband terms: c0 + c1/s + c2/s^2 (solved linearly per alpha).
    """
    # Precompute template on a dense s-grid for accurate interpolation
    s_dense = np.linspace(30.0, 220.0, 2001)
    xi_temp_dense = xi_template_LCDM(k, s_dense)

    invs  = 1.0/np.clip(s, 1e-6, None)
    invs2 = invs**2

    def chi2_for(alpha):
        st = s * alpha
        # interpolate template at dilated separations
        xi_t = np.interp(st, s_dense, xi_temp_dense, left=xi_temp_dense[0], right=xi_temp_dense[-1])
        X = np.column_stack([xi_t, np.ones_like(s), invs, invs2])  # [A*xi_t + B0 + B1/s + B2/s^2]
        beta, *_ = np.linalg.lstsq(X, xi_mock, rcond=None)
        resid = xi_mock - X @ beta
        return float(np.dot(resid, resid))

    # Golden-section search (no heavy grids)
    a, b = 0.94, 1.06
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr*(b - a)
    d = a + gr*(b - a)
    fc = chi2_for(c)
    fd = chi2_for(d)
    for _ in range(40):  # ~40 iters → ~1e-6 bracket; cheap
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr*(b - a)
            fc = chi2_for(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr*(b - a)
            fd = chi2_for(d)
    alpha_hat = (a + b) / 2
    return {"alpha_hat": float(alpha_hat)}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--omega", type=float, default=0.12)
    ap.add_argument("--eps_bao", type=float, default=0.006)
    ap.add_argument("--width", type=float, default=np.pi/16)
    a=ap.parse_args()
    k, s, xi = make_mock_xi(a.tau, a.omega, a.eps_bao, a.width)
    out = fit_alpha_only_fast(s, xi, k)
    print(json.dumps({"alpha_hat": out["alpha_hat"], "eps_bao": a.eps_bao}, indent=2))

