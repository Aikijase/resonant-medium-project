#!/usr/bin/env python3
"""
Chladni-Cosmology Toy (stable integrator)
-----------------------------------------
- 2D damped, driven membrane with fixed edges (u=0 on boundary)
- Stable velocity-position (leapfrog-like) update with Rayleigh damping
- Optional hyperviscosity term to kill high-frequency blowups
- Frequency can switch mid-run to emulate a "mode shift"

Outputs:
  figs/pattern_*.png
  outputs/dominant_spacing.csv
  outputs/spacing_vs_time.png
  outputs/run_config.json
"""

import argparse, os, json, numpy as np
import matplotlib.pyplot as plt

def radial_power_spectrum(img):
    # Normalize to avoid FFT overflow and remove DC
    X = np.nan_to_num(img - np.nanmean(img))
    std = np.nanstd(X)
    if std > 0: X = X / std
    F = np.fft.fftshift(np.fft.fft2(X))
    P = np.abs(F)**2
    ny, nx = X.shape
    cy, cx = ny//2, nx//2
    Y, Xc = np.ogrid[:ny, :nx]
    r = np.sqrt((Y-cy)**2 + (Xc-cx)**2).astype(np.int32)
    bc = np.bincount(r.ravel())
    bc[bc==0] = 1
    Pr = np.bincount(r.ravel(), P.ravel()) / bc
    return np.arange(Pr.size), Pr

def laplacian(Z, dx):
    L = np.zeros_like(Z)
    L[1:-1,1:-1] = (Z[1:-1,2:] + Z[1:-1,:-2] + Z[2:,1:-1] + Z[:-2,1:-1] - 4.0*Z[1:-1,1:-1]) / (dx*dx)
    return L

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--save-every', type=int, default=200)
    ap.add_argument('--c', type=float, default=1.0)          # wave speed
    ap.add_argument('--dt', type=float, default=None)        # time step
    ap.add_argument('--dx', type=float, default=1.0)         # spatial step
    ap.add_argument('--gamma', type=float, default=0.05)     # Rayleigh damping on velocity
    ap.add_argument('--nu', type=float, default=0.001)       # hyperviscosity on velocity (0 = off)
    ap.add_argument('--A', type=float, default=0.25)         # drive amplitude
    ap.add_argument('--freq1', type=float, default=1.8)
    ap.add_argument('--freq2', type=float, default=2.3)
    ap.add_argument('--switch-step', type=int, default=1500) # None -> no switch
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs('figs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    N, dx, c = args.N, args.dx, args.c
    gamma, nu, A = args.gamma, args.nu, args.A
    steps, save_every = args.steps, args.save_every
    freq1, freq2 = args.freq1, args.freq2
    switch_step = args.switch_step

    # CFL safety for 2D wave eq: dt <= dx / (sqrt(2)*c). Choose conservative factor.
    if args.dt is None:
        dt = 0.25 * dx / (np.sqrt(2.0) * max(1e-12, c))
    else:
        dt = args.dt

    # Fields
    u  = np.zeros((N, N), dtype=np.float64)   # displacement
    v  = np.zeros_like(u)                     # velocity

    # Tiny symmetry-breaking noise
    u += 1e-4*np.random.randn(N, N)

    def omega(step):
        return freq2 if (switch_step is not None and step >= switch_step) else freq1

    spacing_log = []

    for step in range(steps):
        t = step * dt
        Om = omega(step)
        drive = A * np.sin(Om * t)  # spatially uniform drive

        # Compute Laplacians
        Lu = laplacian(u, dx)

        # Velocity update (Rayleigh damped + hyperviscosity on v)
        v = (1.0 - gamma*dt) * v + dt*(c*c * Lu + drive)
        if nu > 0.0:
            Lv = laplacian(v, dx)
            v += dt * nu * Lv

        # Position update
        u = u + dt * v

        # Enforce fixed boundaries
        u[0,:]=u[-1,:]=u[:,0]=u[:,-1]=0.0
        v[0,:]=v[-1,:]=v[:,0]=v[:,-1]=0.0

        # Save figures and metrics
        if (step % save_every == 0) or (step == steps-1):
            U = np.nan_to_num(u.copy())
            # Robust color scale to avoid overflow warnings
            lo, hi = np.percentile(U, 1), np.percentile(U, 99)
            if not np.isfinite(lo): lo = -1.0
            if not np.isfinite(hi): hi = 1.0
            if hi <= lo: hi = lo + 1e-6

            fig = plt.figure(figsize=(6,5), dpi=120)
            im = plt.imshow(U, origin='lower', interpolation='nearest', vmin=lo, vmax=hi)
            plt.title(f"Step {step}  Ω={Om:.3f}")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(f"figs/pattern_{step:06d}.png")
            plt.close(fig)

            # Dominant spacing via radial spectrum
            r, Pr = radial_power_spectrum(U)
            if Pr.size > 5:
                k_peak = int(np.argmax(Pr[5:]) + 5)
            else:
                k_peak = int(np.argmax(Pr))
            spacing = (N / max(1, k_peak)) * dx
            spacing_log.append((step, int(k_peak), float(spacing)))

    # Save logs
    import csv
    with open('outputs/dominant_spacing.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step','k_peak','spacing_est_dx_units'])
        w.writerows(spacing_log)

    with open('outputs/run_config.json', 'w') as f:
        json.dump({
            'N': N, 'steps': steps, 'save_every': save_every,
            'c': c, 'dt': dt, 'dx': dx, 'gamma': gamma, 'nu': nu,
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
