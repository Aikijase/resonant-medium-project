#!/usr/bin/env python3
"""
Phase-5: Damping toy ODE for resonance stability intuition.

ODE:
  x'' + γ x' + ω0^2 x = 0,   with ω0 = f * ω_ref  (ω_ref=1 by default)

We simulate a few representative (γ, f) points:
- best (γ0, f0)
- ± small steps in γ and f

Outputs:
  outputs/phase5/damping_runs.json        (summary metrics)
  plots/phase5/phase5_damping_evolution.png  (time series)

CLI:
  --g0 1.56 --f0 2.60
  --gstep 0.04 --fstep 0.08
  --tmax 10.0 --dt 0.001
"""
import json, math, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_JSON = "outputs/phase5/damping_runs.json"
PLOT     = "plots/phase5/phase5_damping_evolution.png"

def simulate(gamma, f, tmax=10.0, dt=0.001, x0=1.0, v0=0.0, wref=1.0):
    # Integrate x'' = -γ x' - ω0^2 x  via semi-implicit Euler (stable for damping)
    n = int(tmax/dt)+1
    t = np.linspace(0.0, tmax, n)
    x = np.zeros(n); v = np.zeros(n)
    x[0]=x0; v[0]=v0
    w0 = f*wref
    for k in range(n-1):
        # v_{k+1} = v_k + dt*(-γ v_k - w0^2 x_k)
        v[k+1] = v[k] + dt*(-gamma*v[k] - (w0*w0)*x[k])
        # x_{k+1} = x_k + dt*v_{k+1}  (semi-implicit)
        x[k+1] = x[k] + dt*v[k+1]
    # metrics
    env = np.maximum.accumulate(np.abs(x[::-1]))[::-1]  # crude decay env
    half_amp = np.max(np.abs(x))*0.5
    try:
        t_half = t[np.where(env<=half_amp)[0][0]]
    except Exception:
        t_half = float("nan")
    zeta = gamma/(2.0*w0)  # damping ratio
    return t, x, {"gamma":gamma,"f":f,"omega0":w0,"zeta":zeta,"t_half":t_half,"x_max":float(np.max(np.abs(x)))}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", type=float, default=1.56)
    ap.add_argument("--f0", type=float, default=2.60)
    ap.add_argument("--gstep", type=float, default=0.04)
    ap.add_argument("--fstep", type=float, default=0.08)
    ap.add_argument("--tmax", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.001)
    args = ap.parse_args()

    Path("outputs/phase5").mkdir(parents=True, exist_ok=True)
    Path("plots/phase5").mkdir(parents=True, exist_ok=True)

    cfgs = [
        ("best", args.g0, args.f0),
        ("g-",   args.g0-args.gstep, args.f0),
        ("g+",   args.g0+args.gstep, args.f0),
        ("f-",   args.g0, args.f0-args.fstep),
        ("f+",   args.g0, args.f0+args.fstep),
    ]

    plt.figure(figsize=(7.5,5.0))
    summaries=[]
    for label,g,f in cfgs:
        t, x, meta = simulate(g, f, tmax=args.tmax, dt=args.dt)
        plt.plot(t, x, label=f"{label} (γ={g:.2f}, f={f:.2f})")
        meta["label"]=label
        summaries.append(meta)

    plt.xlabel("t (arb.)")
    plt.ylabel("x(t)")
    plt.title("Damped Oscillator Evolution (Phase-5 Toy Model)")
    plt.legend(loc="upper right", ncol=1, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=140)

    with open(OUT_JSON,"w") as f:
        json.dump({"runs": summaries}, f, indent=2)

    print(f"Wrote {OUT_JSON}\nWrote {PLOT}")

if __name__ == "__main__":
    main()
