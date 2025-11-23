#!/usr/bin/env python3
"""
Lightweight Kuramoto-like simulator with per-edge coupling K_ij scaled by
alpha * Kmin(omega_ref), where omega_ref = min(omega_i, omega_j) relative to
a fitted Phase-10 boundary curve (tongue_master_wide.fit.json).

Metrics:
- R_mean: time-averaged global order parameter (|mean e^{iθ}|).
- pair_coh: mean cos(θ_j - θ_i) over edges (last samples).
- K_edge: mean K_ij used (so you can see what the network "spent").
- phase_span: end-to-end spread of final unwrapped phase (≈ drift slope).
- circ_var: circular variance across time (0=perfectly locked).
"""
import json, math
import numpy as np

def load_fit(p="outputs/phase10/tongue_master_wide.fit.json"):
    with open(p) as f:
        j = json.load(f)
    return dict(a=j["a"], b=j["b"], c=j["c"], x0=j.get("x0", 2.8))

def Kmin(omega, fit):
    x = omega - fit["x0"]
    k = fit["a"] + fit["b"]*x + fit["c"]*x*x
    return float(max(0.0, k))

def build_adjacency(topology, N, shortcuts=0, rng=None):
    A = np.zeros((N, N), dtype=float)
    if topology == "ring":
        for i in range(N):
            A[i, (i-1) % N] = 1.0
            A[i, (i+1) % N] = 1.0
    elif topology == "chain":
        for i in range(N-1):
            A[i, i+1] = A[i+1, i] = 1.0
    elif topology == "prism6":
        if N != 6:
            raise ValueError("prism6 expects N=6")
        tri = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]
        rungs = [(0,3),(1,4),(2,5)]
        for u,v in tri + rungs:
            A[u,v] = A[v,u] = 1.0
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # optional random shortcuts (small-world)
    if shortcuts > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        cand = [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j] == 0]
        rng.shuffle(cand)
        for u, v in cand[:shortcuts]:
            A[u, v] = A[v, u] = 1.0

    return A

def order_param(theta):
    z = np.exp(1j*theta).mean()
    return abs(z)

def run_sim(topology="prism6",
            w=None,
            alpha=1.0,
            steps=20000,
            dt=0.01,
            noise=0.0,
            seed=0,
            burn_in=2000,
            sample_every=10,
            tail_samples=500,
            fit_path="outputs/phase10/tongue_master_wide.fit.json",
            shortcuts=0):

    rng = np.random.default_rng(seed)
    if w is None:
        N = 6 if topology == "prism6" else 8
        w = np.linspace(2.6, 3.0, N)
    else:
        w = np.asarray(w)
        N = len(w)

    A = build_adjacency(topology, N, shortcuts=shortcuts, rng=rng)
    fit = load_fit(fit_path)

    # Precompute edge-wise K_ij
    Kij = np.zeros_like(A)
    edges = np.argwhere(A > 0.0)
    for i, j in edges:
        kref = min(Kmin(w[i], fit), Kmin(w[j], fit))
        Kij[i, j] = alpha * kref

    theta = rng.uniform(0, 2*np.pi, size=N)
    Rs = []
    pair_vals = []
    tail_thetas = []
    kept = 0
    sigma = math.sqrt(2*noise*dt) if noise > 0 else 0.0

    for t in range(steps):
        dtheta = w.copy()
        for i in range(N):
            if A[i].sum() > 0:
                s = 0.0
                for j in range(N):
                    if A[i, j] > 0:
                        s += Kij[i, j] * math.sin(theta[j] - theta[i])
                dtheta[i] += s

        if noise > 0:
            dtheta += rng.normal(0.0, 1.0, size=N) * sigma

        theta = (theta + dt * dtheta) % (2 * math.pi)

        if t >= burn_in and (t - burn_in) % sample_every == 0:
            Rs.append(order_param(theta))
            csum = 0.0
            for i, j in edges:
                csum += math.cos(theta[j] - theta[i])
            pair_vals.append(csum / max(1, len(edges)))

            tail_thetas.append(theta.copy())
            kept += 1
            if kept > tail_samples:
                tail_thetas.pop(0)
                kept -= 1

    # Final metrics
    R_mean = float(np.mean(Rs)) if Rs else 0.0
    pair_coh = float(np.mean(pair_vals)) if pair_vals else 0.0
    K_edge = float(Kij[A > 0.0].mean()) if (A > 0.0).any() else 0.0

    if tail_thetas:
        TH = np.stack(tail_thetas, axis=0)
        THu = np.unwrap(TH, axis=0)
        last = THu[-1]
        phase_span = float(last.max() - last.min())
        z = np.exp(1j * TH)
        circ_var = float(1 - abs(z.mean(axis=0)).mean())
    else:
        phase_span = 0.0
        circ_var = 0.0

    return dict(R_mean=R_mean,
                pair_coh=pair_coh,
                K_edge=K_edge,
                phase_span=phase_span,
                circ_var=circ_var)

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", choices=["ring", "chain", "prism6"], default="prism6")
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn_in", type=int, default=2000)
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--tail", type=int, default=500)
    ap.add_argument("--fit", default="outputs/phase10/tongue_master_wide.fit.json")
    ap.add_argument("--shortcuts", type=int, default=0)
    args = ap.parse_args()

    # default freqs
    w = np.linspace(2.6, 3.0, args.N)

    res = run_sim(topology=args.topology, w=w, alpha=args.alpha,
                  steps=args.steps, dt=args.dt, noise=args.noise,
                  seed=args.seed, burn_in=args.burn_in,
                  sample_every=args.sample_every, tail_samples=args.tail,
                  fit_path=args.fit, shortcuts=args.shortcuts)
    print(json.dumps(res))
