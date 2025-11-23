#!/usr/bin/env python3
# Phase 9 — URO Simulator (no-SciPy RK4 integrator)
import argparse, os, sys, datetime, csv, math
import numpy as np
try:
    import yaml
except ImportError:
    print("Missing 'pyyaml' (pip install pyyaml)", file=sys.stderr); sys.exit(1)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Matplotlib not available:", e, file=sys.stderr); sys.exit(1)

def ensure_parent(p):
    d=os.path.dirname(p); 
    os.makedirs(d, exist_ok=True) if d else None
def nowtag(): 
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def build_xstar_fn(cfg):
    t= (cfg or {}).get("type","constant").lower()
    if t=="constant":
        v=float(cfg.get("value",0.0)); return lambda tt: v
    if t=="sinusoid":
        A=float(cfg.get("A",0.0)); om=float(cfg.get("omega",1.0))
        phi=float(cfg.get("phi",0.0)); off=float(cfg.get("offset",0.0))
        return lambda tt: off + A*np.sin(om*tt+phi)
    if t=="step":
        t0=float(cfg.get("t0",0.0)); v0=float(cfg.get("value0",0.0)); v1=float(cfg.get("value1",1.0))
        return lambda tt: v0 if tt<t0 else v1
    raise ValueError("bad target.type")

class BetaAdapt:
    def __init__(self, beta, k=0.0, target_err=0.1, lo=1e-6, hi=1e6):
        self.base=float(beta); self.k=float(k); self.t=float(target_err); self.lo=float(lo); self.hi=float(hi)
    @staticmethod
    def from_cfg(beta,c):
        if not c: return BetaAdapt(beta)
        return BetaAdapt(beta, c.get("k",0.0), c.get("target_error",0.1), c.get("limit_lo",1e-6), c.get("limit_hi",1e6))
    def eff(self, err):
        if self.k<=0: return self.base
        s=abs(err)/max(self.t,1e-12)-1.0
        be=self.base*(1.0+self.k*np.tanh(s))
        return float(np.clip(be,self.lo,self.hi))

def rhs_builder(alpha, beta_adapt, gamma, xstar_fn, F, omg, noise_std=0.0, seed=None):
    rng = np.random.default_rng(seed) if (noise_std and noise_std>0) else None
    def noise():
        return rng.normal(0.0, noise_std) if rng is not None else 0.0
    if alpha<=0: raise ValueError("alpha must be >0")
    def rhs(t,y):
        x,v=y; xstar=xstar_fn(t); err=x-xstar
        be=beta_adapt.eff(err)
        forc= F*np.sin(omg*t) if F!=0.0 else 0.0
        a = (-be*v - gamma*err + forc + noise())/alpha
        return np.array([v,a], dtype=float)
    return rhs

def rk4(f, t0, t1, y0, nstep):
    t = np.linspace(t0,t1,nstep)
    y = np.zeros((nstep, len(y0)), dtype=float)
    y[0]=y0
    for i in range(nstep-1):
        h = t[i+1]-t[i]
        k1=f(t[i], y[i])
        k2=f(t[i]+0.5*h, y[i]+0.5*h*k1)
        k3=f(t[i]+0.5*h, y[i]+0.5*h*k2)
        k4=f(t[i]+h,     y[i]+h*k3)
        y[i+1]=y[i]+(h/6.0)*(k1+2*k2+2*k3+k4)
    return t,y

def main():
    ap=argparse.ArgumentParser(description="URO Simulator (no-SciPy RK4)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-prefix", default=os.path.join("outputs","phase9",f"run_{nowtag()}"))
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args=ap.parse_args()

    with open(args.config,"r") as f: cfg=yaml.safe_load(f) or {}
    sim=cfg.get("sim",{}); dom=cfg.get("domain","unspecified")
    params=cfg.get("params",{}); drive=cfg.get("drive",{})
    target=cfg.get("target",{}); noise=cfg.get("noise",{}); adapt=cfg.get("adaptation",{})

    alpha=float(params.get("alpha",1.0))
    beta =float(params.get("beta",0.2))
    gamma=float(params.get("gamma",1.0))
    F=float(drive.get("F",0.0)); omg=float(drive.get("omega",1.0))

    t0=float(sim.get("t0",0.0))
    t1=float(sim.get("t_max",40.0 if args.tmax is None else args.tmax))
    n =int(sim.get("n_steps",4000 if args.steps is None else args.steps))
    x0=float(sim.get("x0",0.0)); v0=float(sim.get("v0",0.0))

    xstar_fn = build_xstar_fn(target)
    noise_std=float(noise.get("std",0.0)); seed=noise.get("seed", None)
    beta_adapt=BetaAdapt.from_cfg(beta, adapt)
    f = rhs_builder(alpha, beta_adapt, gamma, xstar_fn, F, omg, noise_std, seed)

    t,y = rk4(f, t0, t1, np.array([x0,v0],dtype=float), n)
    x=y[:,0]; v=y[:,1]
    xstar=np.array([xstar_fn(tt) for tt in t],dtype=float)
    err=x-xstar
    beta_eff=np.array([beta_adapt.eff(e) for e in err], dtype=float)
    energy=0.5*alpha*v*v + 0.5*gamma*err*err

    out=args.out_prefix
    ensure_parent(out+".csv")
    with open(out+".csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["t","x","v","x_star","err","beta_eff","energy"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.12g}", f"{x[i]:.12g}", f"{v[i]:.12g}", f"{xstar[i]:.12g}", f"{err[i]:.12g}", f"{beta_eff[i]:.12g}", f"{energy[i]:.12g}"])
    print(f"[phase9-nosci] CSV saved: {out}.csv")

    # plots
    plt.figure(figsize=(9,4.8)); plt.plot(t,x,label="x(t)"); plt.plot(t,xstar,'--',label="x*(t)")
    plt.xlabel("t"); plt.ylabel("State"); plt.title(f"URO Time Series — {dom}")
    plt.legend(); plt.tight_layout(); plt.savefig(out+"_timeseries.png",dpi=150); plt.close()
    print(f"[phase9-nosci] Plot saved: {out}_timeseries.png")

    plt.figure(figsize=(5.6,5.6)); plt.plot(x,v)
    plt.xlabel("x"); plt.ylabel("v = ẋ"); plt.title(f"URO Phase Portrait — {dom}")
    plt.tight_layout(); plt.savefig(out+"_phase.png",dpi=150); plt.close()
    print(f"[phase9-nosci] Plot saved: {out}_phase.png")

    # quick metrics
    mid=len(t)//2
    p2p=np.max(x[mid:])-np.min(x[mid:]) if mid<len(x) else float("nan")
    rms=np.sqrt(np.mean(err[mid:]**2)) if mid<len(err) else float("nan")
    print(f"[phase9-nosci] Metrics (post-mid): peak_to_peak={p2p:.6g}  rms_err={rms:.6g}")

if __name__=="__main__":
    main()
