#!/usr/bin/env python3
"""
tools/magnet_echo_demo.py  —  Wave+memory+magnet (stable, oscillatory)

Key changes vs previous:
- Second-order dynamics (y,v) → real oscillations and peaks.
- Longer, near-resonant drive; Gaussian spatial footprint.
- Sample spatial-average of |y| to avoid node cancellation.
- Keeps magnet coupling + anisotropic advection.
- Numerically stable defaults for Chromebook Penguin.

Outputs:
  outputs/phaseXX/echo_spectrum_tauM*.png
  outputs/phaseXX/magnet_field_tauM*.png
  outputs/phaseXX/echo_field_tauM*.png
  outputs/phaseXX/magnet_echo_summary.csv
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import welch

# --------------------------- CONFIG --------------------------- #
OUTDIR = "outputs/phaseXX"; os.makedirs(OUTDIR, exist_ok=True)

# Grid / time
NX, NY = 96, 96
DT = 0.01
STEPS = 8000           # ~80 s of simulated time
SAVE_EVERY = 20        # downsample for spectrum
SEED = 7

# Wave / echo params (second-order system)
C = 1.0                # wave speed
OMEGA0 = 1.6           # natural frequency (try close to drive)
ZETA = 0.04            # damping ratio (~2*ZETA*OMEGA0 term)
BETA = 0.10            # memory feedback strength (EMA of y)

# Memory times (will be swept)
TAU_M_LIST = [2.0, 5.0, 10.0, 20.0]

# Magnet (vector memory) params
SIGMA = 0.30           # source from |∇y|
GAMMA_M = 0.10         # saturation
D_M = 0.03             # diffusion
ALPHA = 0.05           # anisotropic advection (M·∇)y
DAMP_M = 0.002         # tiny linear damping

# Drive (sinusoid, spatial Gaussian)
DRIVE_STEPS = 2500
DRIVE_OMEGA = 1.5      # close to OMEGA0 to excite resonance
DRIVE_AMP = 0.030
DRIVE_SIGMA = 5.0      # Gaussian width (pixels)

# Spectrum config
WELCH_NPERSEG = 512
EPS = 1e-9
DPI = 150

# ------------------------ UTILITIES --------------------------- #
def lap(A):
    return (np.roll(A,1,0)+np.roll(A,-1,0)+np.roll(A,1,1)+np.roll(A,-1,1)-4*A)

def grad(A):
    gx = 0.5*(np.roll(A,-1,0)-np.roll(A,1,0))
    gy = 0.5*(np.roll(A,-1,1)-np.roll(A,1,1))
    return gx, gy

def unit(vx, vy):
    m = np.sqrt(vx*vx+vy*vy)+EPS
    return vx/m, vy/m, m

def ema(prev, new, tau, dt):
    if tau <= 0: return new.copy()
    return prev + (dt/tau)*(new - prev)

def halfmax_Q(f, P):
    if len(f) < 4 or np.all(P <= 0): return 0.0, 0.0, 0.0
    i0 = int(np.argmax(P)); pf = float(f[i0]); pp = float(P[i0])
    if pp <= 0: return pf, 0.0, 0.0
    half = pp*0.5
    iL = i0
    while iL > 0 and P[iL] > half: iL -= 1
    iR = i0
    while iR < len(P)-1 and P[iR] > half: iR += 1
    bw = float(f[iR]-f[iL]) if (iR > iL) else 0.0
    Q = (pf/bw) if bw > 0 else 0.0
    return pf, bw, Q

def stable_clip(a, lim=1e6):
    return np.clip(a, -lim, lim)

# ---------------------- DRIVE MASK ---------------------------- #
def gaussian_drive(nx, ny, sigma_pix):
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = nx//2, ny//2
    R2 = (X-cx)**2 + (Y-cy)**2
    G = np.exp(-R2/(2.0*sigma_pix**2))
    # normalize so sum(G)=1 (drive amplitude is in DRIVE_AMP)
    G /= (G.sum() + EPS)
    return G

DRIVE_MASK = gaussian_drive(NX, NY, DRIVE_SIGMA)

# ---------------------- ONE SIMULATION ------------------------ #
def run_for_tau(tau_m):
    rng = np.random.default_rng(SEED)
    # Fields
    y = 1e-3 * rng.standard_normal((NX, NY))   # displacement
    v = np.zeros_like(y)                        # velocity
    ym = np.zeros_like(y)                       # EMA memory
    Mx = np.zeros_like(y)                       # magnet vector
    My = np.zeros_like(y)

    series = []  # time series = spatial mean of |y| (avoid nodes)

    two_z_om0 = 2.0*ZETA*OMEGA0

    for t in range(STEPS):
        # Driving
        if t < DRIVE_STEPS:
            y += DRIVE_AMP * np.sin(DRIVE_OMEGA * t * DT) * DRIVE_MASK

        # Spatial ops
        lap_y = lap(y)
        gx, gy = grad(y)
        nxg, nyg, gmag = unit(gx, gy)

        # Magnet update (Euler)
        Mx += DT * (
            -Mx/max(tau_m, DT) + D_M*lap(Mx) + SIGMA*(gmag*nxg)
            - GAMMA_M*(Mx*Mx+My*My)*Mx - DAMP_M*Mx
        )
        My += DT * (
            -My/max(tau_m, DT) + D_M*lap(My) + SIGMA*(gmag*nyg)
            - GAMMA_M*(Mx*Mx+My*My)*My - DAMP_M*My
        )
        Mx = stable_clip(Mx, 1e3); My = stable_clip(My, 1e3)

        # Memory update
        ym = ema(ym, y, tau=tau_m, dt=DT)

        # Advection from magnets
        adv = ALPHA * (Mx*gx + My*gy)

        # Second-order wave:  y' = v
        #                     v' = c^2∇^2 y - 2ζω0 v - ω0^2 y + β ym + adv
        v += DT * (C*C*lap_y - two_z_om0*v - OMEGA0*OMEGA0*y + BETA*ym + adv)
        y += DT * v

        # Mild global damping/clip for numerical hygiene
        y *= 0.999
        y = np.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        v = np.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)

        if (t % SAVE_EVERY) == 0:
            series.append(np.mean(np.abs(y)))

    return np.asarray(series, float), Mx, My, y

# ---------------------- RUN & SAVE ---------------------------- #
def main():
    rows = []
    for tau in TAU_M_LIST:
        series, Mx, My, y = run_for_tau(tau)

        # Spectrum
        fs = 1.0/(DT*SAVE_EVERY)
        nperseg = min(WELCH_NPERSEG, max(128, len(series)//2))
        x = series - series.mean()
        f, Pxx = welch(x, fs=fs, nperseg=nperseg)
        pf, bw, Q = halfmax_Q(f, Pxx)

        # Plots
        fig, ax = plt.subplots(figsize=(5,4), dpi=DPI)
        ax.semilogy(f, Pxx + 1e-16)
        ax.set_title(f"Echo Spectrum (τₘ={tau:g})")
        ax.set_xlabel("Frequency"); ax.set_ylabel("Power")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"echo_spectrum_tauM{int(tau)}.png"))
        plt.close(fig)

        skip = 5
        X, Y = np.meshgrid(np.arange(0,NX,skip), np.arange(0,NY,skip), indexing="ij")
        U, V = Mx[::skip,::skip], My[::skip,::skip]
        fig, ax = plt.subplots(figsize=(5,5), dpi=DPI)
        ax.quiver(Y, X, V, U, scale=60)
        ax.set_title(f"Magnet Field Snapshot (τₘ={tau:g})")
        ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"magnet_field_tauM{int(tau)}.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5,5), dpi=DPI)
        im = ax.imshow(y.T, origin="lower", interpolation="nearest")
        ax.set_title(f"Echo Field |y| Snapshot (τₘ={tau:g})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"echo_field_tauM{int(tau)}.png"))
        plt.close(fig)

        rows.append(dict(tauM=tau, samples=len(series), fs=fs,
                         peak_freq=pf, bandwidth_halfmax=bw, Q_factor=Q))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "magnet_echo_summary.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote summary → {csv_path}")
    print(f"Figures in → {OUTDIR}/")

if __name__ == "__main__":
    main()
