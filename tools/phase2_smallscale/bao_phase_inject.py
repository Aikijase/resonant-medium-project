import numpy as np
import json
from tools.phase2_smallscale.halo_mass_function import linear_pk
from tools.phase2_smallscale.utils_node_models import resonant_modifier_phase_smeared
from tools.phase2_smallscale.bao_phase import dewiggle, bao_phase_shift

def window_bao(k, klo=0.05, khi=0.30, edge=0.02):
    """Smooth BAO window: 0 outside band, 1 inside, eased edges."""
    def s(x): x=np.clip(x,0,1); return x*x*(3-2*x)
    w=np.zeros_like(k, float)
    up  = (k>=klo-edge)&(k<klo);   w[up]  = s((k[up]-(klo-edge))/edge)
    mid = (k>=klo)&(k<=khi);       w[mid] = 1.0
    dn  = (k>khi)&(k<=khi+edge);   w[dn]  = 1 - s((k[dn]-khi)/edge)
    return w

def apply_bao_phase_warp(k, P_in, epsilon=0.006, klo=0.05, khi=0.30, edge=0.02):
    """Apply a small BAO-localized phase warp safely."""
    Pnw, W0 = dewiggle(P_in, k)
    w = window_bao(k, klo, khi, edge)
    kshift = k + epsilon * w
    kshift = np.clip(kshift, k[0], k[-1])
    W_shift = np.interp(kshift, k, W0)
    return Pnw * W_shift

def make_pk_with_bao_phase(tau, omega, eps_bao, width=np.pi/16):
    k = np.logspace(-3, 1.0, 4000)
    P0 = linear_pk(k)
    mod_res = resonant_modifier_phase_smeared(k, tau, omega, n_phase=9, width=width)
    P_res = P0 * mod_res
    P_bao = apply_bao_phase_warp(k, P_res, epsilon=eps_bao)
    return k, P0, P_res, P_bao

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--omega", type=float, default=0.12)
    ap.add_argument("--eps_bao", type=float, default=0.006)
    ap.add_argument("--width", type=float, default=np.pi/16)
    a = ap.parse_args()

    k, P0, P_res, P_bao = make_pk_with_bao_phase(a.tau, a.omega, a.eps_bao, a.width)

    # measure BAO phase shift vs LCDM baseline
    _, W0 = dewiggle(P0, k)
    _, Wb = dewiggle(P_bao, k)
    eps, r = bao_phase_shift(k, W0, Wb)

    print(json.dumps({
        "tau": a.tau,
        "omega": a.omega,
        "eps_input": a.eps_bao,
        "epsilon_hMpc": eps,
        "corr": r
    }, indent=2))
