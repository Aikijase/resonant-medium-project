#!/usr/bin/env python3
"""
Chladni-Cosmology Toy: 2D Driven Membrane with Mode Shifts
----------------------------------------------------------
A single-file simulation you can run from the terminal.

Concept
-------
- Treat spacetime like a "membrane" with fixed edges (u=0 at boundaries).
- Drive it with a sinusoidal force of frequency Ω(t) that you can change over time.
- Damping lets the system settle into standing-wave patterns (nodes/antinodes).
- Save snapshots and estimate the dominant node spacing from the 2D FFT.

What you get
------------
- PNG frames in ./figs (pattern snapshots)
- A CSV ./outputs/dominant_spacing.csv with time index, dominant k, and estimated spacing
- A plot ./outputs/spacing_vs_time.png of the dominant spacing vs. time
- A JSON ./outputs/run_config.json recording your parameters

Quick start (default small run)
-------------------------------
$ python3 chladni_cosmo_toy.py

Custom run (more steps/grid, two-stage frequency to emulate a "mode shift"):
$ python3 chladni_cosmo_toy.py --N 192 --steps 6000 --save-every 300 \
    --freq1 1.6 --freq2 2.2 --switch-step 3000 --gamma 0.02 --c 1.0 --A 0.8

Arguments
---------
--N              Grid size (NxN). Default 128
--steps          Number of time steps. Default 3000
--save-every     Save a PNG every k steps. Default 200
--c              Wave speed parameter. Default 1.0
--dt             Time step; if omitted, chosen for stability. Default None
--dx             Grid spacing; affects wavelength units. Default 1.0
--gamma          Damping coefficient (>=0). Default 0.03
--A              Drive amplitude. Default 0.6
--freq1          Drive frequency before switch. Default 1.8
--freq2          Drive frequency after switch. Default 2.3
--switch-step    Step to switch freq1 -> freq2 (None = no switch). Default 1500
--seed           RNG seed for tiny initial noise. Default 0

Notes
-----
- Units are arbitrary but self-consistent (dx sets spatial unit; dt set for stability).
- Fixed-edge boundary (Dirichlet): u=0 at borders, approximating a clamped plate.
- This is a pedagogical toy, not a precise PDE solver for plates; it’s enough to visualize modes.
"""

import argparse, os, json, numpy as np
import matplotlib.pyplot as plt

def radial_power_spectrum(img):
    """
    Compute a simple isotropic (radial) power spectrum of a 2D image using FFT.
    Returns (r, P_r) where r is radial frequency index and P_r is mean power in that ring.
    """
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F)**2
    ny, nx = img.shape
    cy, cx = ny//2, nx//2
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((Y-cy)**2 + (X-cx)**2)
    r = r.astype(np.int32)
    maxr = r.max()
    P_r = np.bincount(r.ravel(), P.ravel()) / np.maximum(1, np.bincount(r.ravel()))
    return np.arange(P_r.size), P_r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--save-every', type=int, default=200)
    ap.add_argument('--c', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--dx', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=0.03)
    ap.add_argument('--A', type=float, default=0.6)
    ap.add_argument('--freq1', type=float, default=1.8)
    ap.add_argument('--freq2', type=float, default=2.3)
    ap.add_argument('--switch-step', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Directories
    os.makedirs('figs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    N = args.N
    dx = args.dx
    c  = args.c
    gamma = args.gamma
    A = args.A
    steps = args.steps
    save_every = args.save_every
    freq1, freq2 = args.freq1, args.freq2
    switch_step = args.switch_step

    # Stability: for 2D wave eq, dt <= dx / (sqrt(2)*c). Use a safety factor.
    if args.dt is None:
        dt = 0.6 * dx / (np.sqrt(2.0) * max(1e-8, c))
    else:
        dt = args.dt

    # Fields
    u     = np.zeros((N, N), dtype=np.float64)
    u_prev= np.zeros_like(u)
    u_next= np.zeros_like(u)

    # Tiny noise to break symmetry
    u += 1e-4*np.random.randn(N, N)

    # Laplacian kernel (5-point stencil)
    def laplacian(Z):
        L = np.zeros_like(Z)
        L[1:-1,1:-1] = (Z[1:-1,2:] + Z[1:-1,:-2] + Z[2:,1:-1] + Z[:-2,1:-1] - 4.0*Z[1:-1,1:-1]) / (dx*dx)
        return L

    # Drive function: Ω(t) switches from freq1 to freq2 at switch_step (if provided)
    def omega(step):
        if (switch_step is not None) and (step >= switch_step):
            return freq2
        return freq1

    spacing_log = []  # (step, dominant_k, spacing_est)

    for step in range(steps):
        t = step * dt
        Om = omega(step)
        drive = A * (np.sin(Om * t))

        Lu = laplacian(u)

        u_next.fill(0.0)
        u_next[1:-1,1:-1] = ((2.0 - 2.0*gamma*dt) * u[1:-1,1:-1]
                             - (1.0 - 2.0*gamma*dt) * u_prev[1:-1,1:-1]
                             + (c*c * dt*dt) * Lu[1:-1,1:-1]
                             + (dt*dt) * drive)

        # Enforce boundary (clamped): u=0 at borders
        u_next[0,:] = 0.0; u_next[-1,:]=0.0; u_next[:,0]=0.0; u_next[:,-1]=0.0

        # Rotate buffers
        u_prev, u = u, u_next

        # Save figure occasionally
        if (step % save_every == 0) or (step == steps-1):
            fig = plt.figure(figsize=(6,5), dpi=120)
            im = plt.imshow(u, origin='lower', interpolation='nearest')
            plt.title(f"Step {step}  Ω={Om:.3f}")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(f"figs/pattern_{step:06d}.png")
            plt.close(fig)

            # Estimate dominant spacing via radial power spectrum
            r, P_r = radial_power_spectrum(u)
            # Ignore r=0 DC; find peak
            if P_r.size > 5:
                k_peak = int(np.argmax(P_r[5:]) + 5)
            else:
                k_peak = int(np.argmax(P_r))
            spacing = (N / max(1, k_peak)) * dx
            spacing_log.append((step, int(k_peak), float(spacing)))

    # Save logs/plots
    import csv, json as _json
    with open('outputs/dominant_spacing.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step','k_peak','spacing_est_dx_units'])
        w.writerows(spacing_log)

    with open('outputs/run_config.json', 'w') as f:
        _json.dump({
            'N': N, 'steps': steps, 'save_every': save_every,
            'c': c, 'dt': dt, 'dx': dx, 'gamma': gamma,
            'A': A, 'freq1': freq1, 'freq2': freq2, 'switch_step': switch_step
        }, f, indent=2)

    # Plot spacing vs time
    if spacing_log:
        steps_arr = np.array([s for s,_,_ in spacing_log], dtype=float)
        spacing_arr = np.array([sp for _,_,sp in spacing_log], dtype=float)
        fig = plt.figure(figsize=(6,4), dpi=120)
        plt.plot(steps_arr, spacing_arr, lw=2)
        plt.xlabel('Step')
        plt.ylabel('Dominant node spacing (dx units)')
        plt.title('Dominant spacing vs. time')
        plt.tight_layout()
        plt.savefig('outputs/spacing_vs_time.png')
        plt.close(fig)

if __name__ == "__main__":
    main()
