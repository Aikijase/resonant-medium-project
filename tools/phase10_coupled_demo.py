#!/usr/bin/env python3
"""
Coupled Resonant Operators — Phase 10 Extension with Sync Metrics + Mismatch.
Simulates two loops with symmetric linear coupling and reports synchrony metrics.

Usage examples:
  # Different seeds + small initial offset (no detuning):
  python tools/phase10_coupled_demo.py --preset neuron --kappa 0.00 --seed1 1 --seed2 2 --init_offset 0.3 --noise 0.01 --prefix k0_off

  # Detune omega for loop 2 and sweep kappa yourself:
  python tools/phase10_coupled_demo.py --preset neuron --omega2 2.6 --kappa 0.10 --seed1 1 --seed2 2 --noise 0.01 --prefix detune

Notes:
- With identical params, seeds, and zero noise, loops can be identical ⇒ sync_index=1 even at kappa=0.
"""

import argparse, os, json, numpy as np
import matplotlib.pyplot as plt
from phase10_resonant_operator import ResonantOperator, PRESETS  # base operator + presets

def run_coupled(op1, op2, kappa=0.1, steps=3000, dt=0.01, burn_in=200):
    x1_hist, x2_hist, v1_hist, v2_hist = [], [], [], []
    for k in range(steps):
        # intrinsic update
        op1.step(dt); op2.step(dt)
        # symmetric linear coupling (mix velocities toward each other)
        dx = (op2.x - op1.x) * kappa
        op1.v += dx * dt
        op2.v -= dx * dt
        if k >= burn_in:
            x1_hist.append(op1.x); v1_hist.append(op1.v)
            x2_hist.append(op2.x); v2_hist.append(op2.v)
    return (np.array(x1_hist), np.array(v1_hist),
            np.array(x2_hist), np.array(v2_hist))

def phase_from_xv(x, v):
    return np.arctan2(v, x)

def wrap_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def sync_metrics(x1, v1, x2, v2):
    phi1 = phase_from_xv(x1, v1)
    phi2 = phase_from_xv(x2, v2)
    dphi = wrap_pi(phi1 - phi2)
    sync_index = np.abs(np.mean(np.exp(1j * dphi)))   # 1 = perfect phase-lock
    mean_abs_phase_diff = float(np.mean(np.abs(dphi)))
    mean_abs_xdiff = float(np.mean(np.abs(x1 - x2)))
    return {
        "sync_index": float(sync_index),
        "mean_abs_phase_diff": mean_abs_phase_diff,
        "mean_abs_xdiff": mean_abs_xdiff
    }, dphi

def plot_sync_ts(t, x1, x2, prefix, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,4.5))
    plt.plot(t, x1, label="Loop 1")
    plt.plot(t, x2, label="Loop 2", alpha=0.75)
    plt.legend(); plt.xlabel("t"); plt.ylabel("x(t)")
    plt.title("Coupled Resonance: Time Series")
    path = os.path.join(outdir, f"{prefix}_sync_timeseries.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_phase_diff(t, dphi, prefix, outdir):
    plt.figure(figsize=(8,4.5))
    plt.plot(t, dphi, lw=1.1)
    plt.xlabel("t"); plt.ylabel("Δφ(t) [rad]")
    plt.title("Phase Difference Over Time (wrapped to ±π)")
    path = os.path.join(outdir, f"{prefix}_phase_diff.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_xcorr(x1, x2, prefix, outdir):
    plt.figure(figsize=(5,5))
    plt.scatter(x1, x2, s=4, alpha=0.6)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title("State Correlation (x1 vs x2)")
    path = os.path.join(outdir, f"{prefix}_x1_vs_x2.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="neuron")
    ap.add_argument("--kappa", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--prefix", type=str, default="coupled_run")
    ap.add_argument("--outdir", type=str, default="outputs/phase10")
    # mismatch knobs
    ap.add_argument("--seed1", type=int, default=1)
    ap.add_argument("--seed2", type=int, default=2)
    ap.add_argument("--init_offset", type=float, default=0.0, help="Add to x2 at start (break identical ICs).")
    ap.add_argument("--v2_offset", type=float, default=0.0, help="Add to v2 at start.")
    ap.add_argument("--omega2", type=float, default=None, help="Detune loop 2 natural frequency.")
    ap.add_argument("--noise", type=float, default=0.01, help="Set same noise for both loops (nonzero breaks identity).")
    args = ap.parse_args()

    # base preset config
    cfg1 = dict(**PRESETS[args.preset])
    cfg2 = dict(**PRESETS[args.preset])
    if args.omega2 is not None:
        cfg2["omega"] = float(args.omega2)
    # ensure noise present (can set to 0 if you want)
    cfg1["noise"] = float(args.noise)
    cfg2["noise"] = float(args.noise)

    # different seeds (critical if noise>0)
    cfg1["seed"] = int(args.seed1)
    cfg2["seed"] = int(args.seed2)

    op1 = ResonantOperator(**cfg1)
    op2 = ResonantOperator(**cfg2)

    # break identical initial conditions
    op2.x += float(args.init_offset)
    op2.v += float(args.v2_offset)

    # run
    x1, v1, x2, v2 = run_coupled(op1, op2, kappa=args.kappa, steps=args.steps, dt=args.dt, burn_in=args.burn_in)
    t = np.arange(len(x1)) * args.dt

    # metrics
    met, dphi = sync_metrics(x1, v1, x2, v2)

    # plots
    os.makedirs(args.outdir, exist_ok=True)
    p_sync = plot_sync_ts(t, x1, x2, args.prefix, args.outdir)
    p_dphi = plot_phase_diff(t, dphi, args.prefix, args.outdir)
    p_xcorr = plot_xcorr(x1, x2, args.prefix, args.outdir)

    # output JSON
    js = dict(
        preset=args.preset,
        kappa=args.kappa,
        steps=args.steps,
        dt=args.dt,
        burn_in=args.burn_in,
        mismatch=dict(seed1=args.seed1, seed2=args.seed2, init_offset=args.init_offset,
                      v2_offset=args.v2_offset, omega2=args.omega2, noise=args.noise),
        metrics=met,
        outputs=dict(sync_timeseries=p_sync, phase_diff=p_dphi, x1_vs_x2=p_xcorr),
        notes="Nonzero noise or initial/parameter mismatch avoids trivial identity. Higher kappa should increase sync_index."
    )
    with open(os.path.join(args.outdir, f"{args.prefix}_summary.json"), "w") as f:
        json.dump(js, f, indent=2)
    print(json.dumps(js, indent=2))

if __name__ == "__main__":
    main()
