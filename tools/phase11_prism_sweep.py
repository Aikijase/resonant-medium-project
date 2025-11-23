#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np
from phase11_prism_sim import run_sim

def make_freqs(N, base=(2.8, 0.0), pattern="alt", span=0.24):
    w0 = base[0]
    if pattern == "alt":
        dw = np.array([(-1)**i for i in range(N)]) * (span/2)
    elif pattern == "grad":
        dw = np.linspace(-span/2, span/2, N)
    elif pattern == "block":
        dw = np.concatenate([np.full(N//2, -span/2), np.full(N - N//2, span/2)])
    else:
        raise ValueError(f"Unknown dw_pattern: {pattern}")
    return w0 + dw

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", choices=["ring","chain","prism6"], default="prism6")
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--dw_patterns", nargs="+", default=["alt","grad","block"])
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--burn_in", type=int, default=2000)
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--tail", type=int, default=500)
    ap.add_argument("--fit", default="outputs/phase10/tongue_master_wide.fit.json")
    ap.add_argument("--shortcuts", type=int, default=0, help="extra long links for small-world")
    ap.add_argument("--out", default="outputs/phase11/data/prism_sweep.csv")
    args = ap.parse_args()

    Path("outputs/phase11/data").mkdir(parents=True, exist_ok=True)

    rows = []
    for pattern in args.dw_patterns:
        w = make_freqs(args.N, base=(2.8, 0.0), pattern=pattern, span=0.24)
        for seed in args.seeds:
            for alpha in args.alpha:
                res = run_sim(topology=args.topology, w=w, alpha=alpha,
                              steps=args.steps, dt=args.dt, noise=args.noise,
                              seed=seed, burn_in=args.burn_in,
                              sample_every=args.sample_every, tail_samples=args.tail,
                              fit_path=args.fit, shortcuts=args.shortcuts)
                rows.append(dict(
                    topology=args.topology,
                    seed=seed,
                    alpha=alpha,
                    dw_pattern=pattern,
                    R_mean=res["R_mean"],
                    pair_coh=res["pair_coh"],
                    K_edge=res["K_edge"],
                    phase_span=res.get("phase_span", 0.0),
                    circ_var=res.get("circ_var", 0.0),
                    shortcuts=args.shortcuts,
                ))
                print(f"[p11] {args.topology} seed={seed} alpha={alpha:.2f} "
                      f"pattern={pattern} shortcuts={args.shortcuts} -> "
                      f"R={res['R_mean']:.3f} pair={res['pair_coh']:.3f} "
                      f"K≈{res['K_edge']:.3f} span≈{res.get('phase_span',0.0):.3f} "
                      f"cvar≈{res.get('circ_var',0.0):.3f}")

    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=[
            "topology","seed","alpha","dw_pattern","R_mean","pair_coh","K_edge",
            "phase_span","circ_var","shortcuts"
        ])
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"[p11] wrote: {args.out}")

if __name__ == "__main__":
    main()
