import numpy as np, json
from tools.phase2_smallscale.halo_mass_function import linear_pk
from tools.phase2_smallscale.utils_node_models import resonant_modifier_phase_smeared
from tools.phase2_smallscale.bao_phase import dewiggle

def tapered_window(k, kmin=0.05, kmax=0.30, edge=0.02):
    w = np.zeros_like(k, float)
    # cosine tapers
    def ramp(x): return 0.5 - 0.5*np.cos(np.clip(x,0,1)*np.pi)
    w += (k>=kmin)&(k<=kmax)
    lo = (k>=kmin-edge)&(k<kmin); w[lo] = ramp((k[lo]-(kmin-edge))/edge)
    hi = (k>kmax)&(k<=kmax+edge); w[hi] = ramp(1-(k[hi]-kmax)/edge)
    return w

def fit_phase_regress(k, W, rdpiv=105.0, kmin=0.05, kmax=0.30, edge=0.02):
    w = tapered_window(k, kmin, kmax, edge)
    sel = w>0
    ks, ws = k[sel], w[sel]
    Y = W[sel]
    # design matrix
    S = np.sin(ks*rdpiv); C = np.cos(ks*rdpiv)
    X = np.column_stack([S, C, np.ones_like(ks), ks, ks**2])
    # weighted least squares
    Wt = np.diag(ws)
    Xt = X.T @ Wt
    beta, *_ = np.linalg.lstsq(Xt @ X, Xt @ Y, rcond=None)
    A, B, c0, c1, c2 = beta
    # residuals & covariance
    Yhat = X @ beta
    res  = Y - Yhat
    dof  = max(1, len(Y) - X.shape[1])
    s2   = (res @ Wt @ res) / dof
    cov  = s2 * np.linalg.pinv(Xt @ X)
    # phase and uncertainty via error propagation
    phi  = np.arctan2(B, A)
    # var(phi) ~ (dphi/dA, dphi/dB) Cov (A,B) (transpose)
    dA, dB = -B/(A*A+B*B), A/(A*A+B*B)
    var_phi = dA**2 * cov[0,0] + dB**2 * cov[1,1] + 2*dA*dB*cov[0,1]
    sigma_phi = float(np.sqrt(max(var_phi, 0.0)))
    # R^2 on window
    ss_res = float((ws*(res**2)).sum())
    ybar   = float((ws*Y).sum()/ws.sum())
    ss_tot = float((ws*((Y-ybar)**2)).sum())
    R2 = 1.0 - ss_res/max(ss_tot, 1e-30)
    return {"phi_rad": float(phi), "sigma_phi": sigma_phi, "R2": float(R2), "A": float(A), "B": float(B)}

def measure_phi_for_model(tau=0.04, omega=0.12, width=np.pi/16, eps_bao=None):
    k = np.logspace(-3, 1.0, 4000)
    P0 = linear_pk(k)
    mod= resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=width)
    P1 = P0*mod
    if eps_bao is not None:
        # simple constant BAO phase warp: resample wiggle at shifted k in BAO band
        Pnw, W = dewiggle(P1, k)
        kmin, kmax, edge = 0.05, 0.30, 0.02
        # smooth window; shift only inside BAO band
        w = np.zeros_like(k)
        w[(k>=kmin)&(k<=kmax)] = 1.0
        kshift = np.clip(k + eps_bao*w, k[0], k[-1])
        Wsh = np.interp(kshift, k, W)
        P1 = Pnw * Wsh
    _, W0 = dewiggle(P0, k)
    _, W1 = dewiggle(P1, k)
    return fit_phase_regress(k, W1)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--omega", type=float, default=0.12)
    ap.add_argument("--width", type=float, default=np.pi/16)
    ap.add_argument("--eps_bao", type=float, default=None)
    a=ap.parse_args()
    out=measure_phi_for_model(a.tau, a.omega, a.width, a.eps_bao)
    print(json.dumps(out, indent=2))
