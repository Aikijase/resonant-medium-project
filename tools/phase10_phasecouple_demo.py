#!/usr/bin/env python3
"""
Phase-coupled resonators (position + velocity + Kuramoto + adaptive freq).
Drop-in extension for Phase 10. No change to phase10_resonant_operator.py.

Usage examples:
  # Small detuning, modest phase/position coupling
  python tools/phase10_phasecouple_demo.py --preset neuron --omega2 2.4 \
    --kv 0.10 --kx 0.10 --Kphi 0.20 --eps 0.00 --noise 0.01 \
    --seed1 1 --seed2 2 --steps 5000 --prefix pcouple_test

  # Harder detuning but with adaptive pull
  python tools/phase10_phasecouple_demo.py --preset neuron --omega2 2.4 \
    --kv 0.10 --kx 0.10 --Kphi 0.30 --eps 0.02 --noise 0.01 \
    --seed1 1 --seed2 2 --steps 5000 --prefix adapt_pull
"""
import argparse, os, json, numpy as np
import matplotlib.pyplot as plt
from phase10_resonant_operator import ResonantOperator, PRESETS

def phase_from_xv(x, v): return np.arctan2(v, x)
def wrap_pi(a): return (a + np.pi) % (2*np.pi) - np.pi

def run(
    op1, op2, *, steps=4000, dt=0.01, burn_in=300,
    kv=0.10, kx=0.00, Kphi=0.00, eps=0.00, adapt_every=1
):
    """kv: velocity coupling, kx: position coupling,
       Kphi: Kuramoto phase coupling, eps: adaptive ω2 pull."""
    x1h, v1h, x2h, v2h, phidh = [], [], [], [], []
    for k in range(steps):
        # intrinsic step
        op1.step(dt); op2.step(dt)

        # coupling terms
        dx = (op2.x - op1.x)
        dv = (op2.v - op1.v)
        phi1 = phase_from_xv(op1.x, op1.v)
        phi2 = phase_from_xv(op2.x, op2.v)
        dphi = wrap_pi(phi2 - phi1)

        # velocity & position pulls
        op1.v += ( kv*dx + kx*dx ) * dt
        op2.v -= ( kv*dx + kx*dx ) * dt

        # Kuramoto phase pull (acts like torque on velocities)
        if Kphi != 0.0:
            op1.v += ( Kphi * np.sin(-dphi) ) * dt  # pull φ1 toward φ2
            op2.v += ( Kphi * np.sin(+dphi) ) * dt  # pull φ2 toward φ1

        # Adaptive frequency pulling on loop 2 (ω2 drifts toward ω1)
        if eps != 0.0 and (k % adapt_every == 0):
            # Move op2's omega opposite the phase error gradient
            # sign chosen so positive dphi (φ2 ahead) reduces ω2, negative increases
            op2.omega += (-eps * np.sin(dphi)) * dt
            op2.omega = float(np.clip(op2.omega, 0.2, 10.0))  # keep sane

        if k >= burn_in:
            x1h.append(op1.x); v1h.append(op1.v)
            x2h.append(op2.x); v2h.append(op2.v)
            phidh.append(dphi)
    return np.array(x1h), np.array(v1h), np.array(x2h), np.array(v2h), np.array(phidh)

def metrics(x1,v1,x2,v2,dphi):
    sync_index = float(np.abs(np.mean(np.exp(1j*dphi))))
    return dict(
        sync_index=sync_index,
        mean_abs_phase_diff=float(np.mean(np.abs(dphi))),
        mean_abs_xdiff=float(np.mean(np.abs(x1-x2)))
    )

def plot_ts(t, x1, x2, out, title, name):
    plt.figure(figsize=(8,4.5))
    plt.plot(t, x1, label="Loop 1")
    plt.plot(t, x2, label="Loop 2", alpha=0.75)
    plt.legend(); plt.xlabel("t"); plt.ylabel("x(t)")
    plt.title(title)
    path=os.path.join(out,f"{name}.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_dphi(t, dphi, out, name):
    plt.figure(figsize=(8,4.5))
    plt.plot(t, dphi, lw=1.1)
    plt.xlabel("t"); plt.ylabel("Δφ(t) [rad]")
    plt.title("Phase Difference (wrapped to ±π)")
    path=os.path.join(out,f"{name}.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_xcorr(x1,x2,out,name):
    plt.figure(figsize=(5,5))
    plt.scatter(x1,x2,s=4,alpha=0.6)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title("State Correlation (x1 vs x2)")
    path=os.path.join(out,f"{name}.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="neuron")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--prefix", type=str, default="pcouple_run")
    ap.add_argument("--outdir", type=str, default="outputs/phase10")

    # mismatch knobs
    ap.add_argument("--omega2", type=float, default=None)
    ap.add_argument("--seed1", type=int, default=1)
    ap.add_argument("--seed2", type=int, default=2)
    ap.add_argument("--init_offset", type=float, default=0.0)
    ap.add_argument("--v2_offset", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.01)

    # coupling strengths
    ap.add_argument("--kv", type=float, default=0.10, help="velocity coupling")
    ap.add_argument("--kx", type=float, default=0.00, help="position coupling")
    ap.add_argument("--Kphi", type=float, default=0.00, help="Kuramoto phase coupling")
    ap.add_argument("--eps", type=float, default=0.00, help="adaptive pull on omega2 per unit time")
    ap.add_argument("--adapt_every", type=int, default=1)

    args=ap.parse_args()
    cfg1=dict(**PRESETS[args.preset]); cfg2=dict(**PRESETS[args.preset])
    if args.omega2 is not None: cfg2["omega"]=float(args.omega2)
    cfg1["noise"]=cfg2["noise"]=float(args.noise)
    cfg1["seed"]=int(args.seed1); cfg2["seed"]=int(args.seed2)

    op1=ResonantOperator(**cfg1)
    op2=ResonantOperator(**cfg2)
    op2.x+=float(args.init_offset); op2.v+=float(args.v2_offset)

    os.makedirs(args.outdir, exist_ok=True)
    x1,v1,x2,v2,dphi = run(
        op1,op2,
        steps=args.steps, dt=args.dt, burn_in=args.burn_in,
        kv=args.kv, kx=args.kx, Kphi=args.Kphi, eps=args.eps, adapt_every=args.adapt_every
    )
    t=np.arange(len(x1))*args.dt
    met=metrics(x1,v1,x2,v2,dphi)

    p_ts = plot_ts(t,x1,x2,args.outdir,"Phase-coupled Resonance", f"{args.prefix}_timeseries")
    p_dp = plot_dphi(t,dphi,args.outdir,f"{args.prefix}_phase_diff")
    p_xc = plot_xcorr(x1,x2,args.outdir,f"{args.prefix}_x1_vs_x2")

    js=dict(
        preset=args.preset, steps=args.steps, dt=args.dt, burn_in=args.burn_in,
        mismatch=dict(omega2=args.omega2, seed1=args.seed1, seed2=args.seed2,
                      init_offset=args.init_offset, v2_offset=args.v2_offset, noise=args.noise),
        coupling=dict(kv=args.kv, kx=args.kx, Kphi=args.Kphi, eps=args.eps, adapt_every=args.adapt_every),
        metrics=met,
        outputs=dict(timeseries=p_ts, phase_diff=p_dp, x1_vs_x2=p_xc),
        notes="Higher kx/Kphi/eps expand the lock region; use gently to avoid instability."
    )
    with open(os.path.join(args.outdir, f"{args.prefix}_summary.json"), "w") as f:
        json.dump(js, f, indent=2)
    print(json.dumps(js, indent=2))

if __name__=="__main__":
    main()
