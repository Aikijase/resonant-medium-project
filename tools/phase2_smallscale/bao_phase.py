import numpy as np
from tools.phase2_smallscale.halo_mass_function import linear_pk
from tools.phase2_smallscale.utils_node_models import resonant_modifier_phase_smeared
try:
    from tools.phase2_smallscale.mod_guard import resonant_modifier_guarded as guarded
except Exception:
    guarded = None

def dewiggle(Pk, k):
    lnk = np.log(k)
    w   = int(max(11, len(k)//50) | 1)
    p   = w // 2
    lnP = np.log(np.clip(Pk, 1e-60, 1e60))
    yy  = np.pad(lnP, (p,p), mode="edge")
    ker = np.ones(w)/w
    lnPnw = np.convolve(yy, ker, mode="valid")
    Pnw = np.exp(lnPnw)
    return Pnw, Pk/Pnw

def bao_phase_shift(k, W_lcdm, W_res, kmin=0.05, kmax=0.3):
    sel = (k>=kmin) & (k<=kmax)
    ks  = k[sel]
    B   = W_res[sel]
    Bn  = (B - B.mean())/B.std()
    eps_grid=np.linspace(-0.02,0.02,401)
    best=(0.0,-1.0)
    A0n_full=(W_lcdm - W_lcdm.mean())/W_lcdm.std()
    for eps in eps_grid:
        kshift = ks + eps
        if kshift[0]<k[0] or kshift[-1]>k[-1]:
            continue
        Ashift = np.interp(kshift, k, A0n_full)
        r = np.corrcoef(Ashift,Bn)[0,1]
        if r>best[1]: best=(eps,r)
    return best

def measure_phase(tau=0.04, omega=0.12, use_guard=False, guard=(4.0,1.0), width=np.pi/16):
    k=np.logspace(-3,0.6,3000)
    P0=linear_pk(k)
    if use_guard and guarded:
        mod = guarded(k,tau,omega,k_guard=guard[0],delta=guard[1],width=width)
    else:
        mod = resonant_modifier_phase_smeared(k,tau,omega,n_phase=9,width=width)
    Pres=P0*mod
    _,W0 = dewiggle(P0,k)
    _,WR = dewiggle(Pres,k)
    eps,r = bao_phase_shift(k,W0,WR)
    return {"epsilon_hMpc":float(eps),"corr":float(r)}

if __name__=="__main__":
    import argparse,json
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau",type=float,default=0.04)
    ap.add_argument("--omega",type=float,default=0.12)
    ap.add_argument("--use_guard",action="store_true")
    ap.add_argument("--k_guard",type=float,default=4.0)
    ap.add_argument("--delta",type=float,default=1.0)
    ap.add_argument("--width",type=float,default=np.pi/16)
    a=ap.parse_args()
    out=measure_phase(a.tau,a.omega,a.use_guard,(a.k_guard,a.delta),a.width)
    print(json.dumps(out,indent=2))
