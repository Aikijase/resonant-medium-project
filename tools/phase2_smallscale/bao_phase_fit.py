import numpy as np, json
from tools.phase2_smallscale.halo_mass_function import linear_pk
from tools.phase2_smallscale.utils_node_models import resonant_modifier_phase_smeared
from tools.phase2_smallscale.bao_phase import dewiggle

def fit_phase(k, W, kmin=0.05, kmax=0.30, rdpiv=105.0, nphi=2001):
    sel=(k>=kmin)&(k<=kmax); ks=k[sel]; Y=W[sel]
    Y=(Y-Y.mean())/np.std(Y)
    phi_grid=np.linspace(-np.pi, np.pi, nphi)
    best=(0.0,-1.0,0.0)
    s=ks*rdpiv
    for phi in phi_grid:
        X=np.sin(s+phi)
        X=(X-X.mean())/np.std(X)
        r=np.dot(X,Y)/len(Y)
        if r>best[1]:
            best=(phi, r, np.sqrt(max(0,r*r)))
    return {"phi_rad": float(best[0]), "corr": float(best[1])}

def measure_phi_for_model(tau=0.04, omega=0.12, width=np.pi/16, eps_bao=None):
    k=np.logspace(-3, 1.0, 4000)
    P0=linear_pk(k)
    mod=resonant_modifier_phase_smeared(k,tau,omega,n_phase=9,width=width)
    P1=P0*mod
    if eps_bao is not None:
        # simple constant phase warp within BAO window (shifts argument)
        kmin,kmax=0.05,0.30
        w=np.zeros_like(k)
        w[(k>=kmin)&(k<=kmax)]=1.0
        kshift=k+eps_bao*w
        kshift=np.clip(kshift,k[0],k[-1])
        # decompose P1, then re-sample wiggles at shifted k
        Pnw, W = dewiggle(P1,k)
        Wsh=np.interp(kshift, k, W)
        P1 = Pnw*Wsh
    _,W0=dewiggle(P0,k)
    _,W1=dewiggle(P1,k)
    return fit_phase(k, W1) | {"ref_phi": fit_phase(k, W0)}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau",type=float,default=0.04)
    ap.add_argument("--omega",type=float,default=0.12)
    ap.add_argument("--width",type=float,default=np.pi/16)
    ap.add_argument("--eps_bao",type=float,default=None)
    a=ap.parse_args()
    out=measure_phi_for_model(a.tau,a.omega,a.width,a.eps_bao)
    # report delta phi relative to LCDM phase=0
    print(json.dumps({
        "phi_rad": out["phi_rad"],
        "corr": out["corr"]
    }, indent=2))
