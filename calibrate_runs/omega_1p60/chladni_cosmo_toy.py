#!/usr/bin/env python3
"""
Chladni-Cosmology Toy (eigen-driven, stable)
--------------------------------------------
- 2D damped, driven membrane with fixed edges (u=0 on boundary)
- Stable velocity–position update (Rayleigh damping + optional hyperviscosity)
- **Eigen-driven forcing**: spatial drive tracks frequency to excite higher modes when Ω rises
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
    X = np.nan_to_num(img - np.nanmean(img))
    std = np.nanstd(X)
    if std > 0: X = X / std
    F = np.fft.fftshift(np.fft.fft2(X))
    P = np.abs(F)**2
    ny, nx = X.shape
    cy, cx = ny//2, nx//2
    Y, Xc = np.ogrid[:ny, :nx]
    r = np.sqrt((Y-cy)**2 + (Xc-cx)**2).astype(np.int32)
    bc = np.bincount(r.ravel()); bc[bc==0] = 1
    Pr = np.bincount(r.ravel(), P.ravel()) / bc
    return np.arange(Pr.size), Pr

def laplacian(Z, dx):
    L = np.zeros_like(Z)
    L[1:-1,1:-1] = (Z[1:-1,2:] + Z[1:-1,:-2] + Z[2:,1:-1] + Z[:-2,1:-1] - 4.0*Z[1:-1,1:-1]) / (dx*dx)
    return L

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=192)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--save-every', type=int, default=300)
    ap.add_argument('--c', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--dx', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=0.05)   # damping on velocity
    ap.add_argument('--nu', type=float, default=0.001)     # hyperviscosity on velocity
    ap.add_argument('--A', type=float, default=0.25)       # drive amplitude
    ap.add_argument('--freq1', type=float, default=1.6)
    ap.add_argument('--freq2', type=float, default=2.2)
    ap.add_argument('--switch-step', type=int, default=3000)
    ap.add_argument('--alpha-k', type=float, default=28.0, help="maps Ω to target k≈alpha_k*Ω")
    ap.add_argument('--drive', choices=['eigen','uniform'], default='eigen')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs('figs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    N, dx, c = args.N, args.dx, args.c
    gamma, nu, A = args.gamma, args.nu, args.A
    steps, save_every = args.steps, args.save_every
    freq1, freq2 = args.freq1, args.freq2
    switch_step, alpha_k = args.switch_step, args.alpha_k

    # CFL safety
    if args.dt is None:
        dt = 0.25 * dx / (np.sqrt(2.0) * max(1e-12, c))
    else:
        dt = args.dt

    # Fields
    u = np.zeros((N, N), dtype=np.float64)
    v = np.zeros_like(u)
    u += 1e-4*np.random.randn(N, N)

    # Coordinates in [0,1]
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing='xy')

    def omega(step):
        return freq2 if (switch_step is not None and step >= switch_step) else freq1

    def drive_mask(step):
        if args.drive == 'uniform':
            return 1.0
        # eigen-driven: choose (m,n) so sqrt(m^2+n^2) ~ k_target
        k_target = max(1, int(round(alpha_k * omega(step))))
        m = max(1, int(round(k_target / np.sqrt(2))))
        n = m
        M = np.sin(np.pi*m*X) * np.sin(np.pi*n*Y)
        M /= (np.max(np.abs(M)) or 1.0)
        return M

    spacing_log = []

    for step in range(steps):
        t = step * dt
        Om = omega(step)
        M = drive_mask(step)
        drive = A * np.sin(Om * t) * M

        Lu = laplacian(u, dx)

        # Velocity update
        v = (1.0 - gamma*dt) * v + dt*(c*c * Lu + drive)
        if nu > 0.0:
            v += dt * nu * laplacian(v, dx)

        # Position update
        u = u + dt * v

        # Fixed boundaries
        u[0,:]=u[-1,:]=u[:,0]=u[:,-1]=0.0
        v[0,:]=v[-1,:]=v[:,0]=v[:,-1]=0.0

        # Save
        if (step % save_every == 0) or (step == steps-1):
            U = np.nan_to_num(u.copy())
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
            'A': A, 'freq1': freq1, 'freq2': freq2, 'switch_step': switch_step,
            'alpha_k': alpha_k, 'drive': args.drive
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
