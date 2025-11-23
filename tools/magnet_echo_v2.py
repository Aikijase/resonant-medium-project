#!/usr/bin/env python3
"""
tools/magnet_echo_v2.py

Echo (finite-memory) + Magnet (vector coherence) demo — fresh build
- Second-order wave dynamics for real oscillations (y, v)
- Exponential memory (EMA) with timescale tau_M (swept)
- Magnet field M = (Mx, My) grows from |∇y|, diffuses, saturates, and biases y via (M·∇)y
- Saves: spectra, quiver, field snapshot, CSV summary, and a 1-page A4 PDF figure

Outputs (under outputs/phaseXX/):
  magnet_echo_summary.csv
  echo_spectrum_tauM*.png
  magnet_field_tauM*.png
  echo_field_tauM*.png
  magnet_echo_figure.pdf        # assembled, paper-ready (Lock → Verify → Style)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch

# --------------------------- CONFIG --------------------------- #
OUTDIR = "outputs/phaseXX"
os.makedirs(OUTDIR, exist_ok=True)

# Speed/quality preset
# For quick preview, set NX=NY=72, STEPS=5000. For nicer figures, 96/8000.
NX, NY   = 96, 96
DT       = 0.01
STEPS    = 8000           # simulated steps (~80s of sim time)
SAVE_EVERY = 20           # downsample for spectrum/series
SEED     = 7              # deterministic

# Wave + Echo parameters
C        = 1.0            # wave speed
OMEGA0   = 1.50           # natural frequency
ZETA     = 0.02           # damping ratio (2*ZETA*OMEGA0 factor on v)
BETA     = 0.18           # memory feedback strength

# Memory sweep (viscosity proxy)
TAU_M_LIST = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]

# Magnet field parameters
SIGMA    = 0.40           # source from |∇y|
GAMMA_M  = 0.08           # saturation
D_M      = 0.015          # diffusion
ALPHA    = 0.08           # anisotropic coupling (M·∇)y
DAMP_M   = 0.002          # tiny linear damping

# Drive: spatial Gaussian near-resonant sinusoid
DRIVE_STEPS = 3000
DRIVE_OMEGA = 1.45        # close to OMEGA0
DRIVE_AMP   = 0.025
DRIVE_SIGMA = 5.0         # pixels

# Spectral analysis
WELCH_NPERSEG = 512
EPS = 1e-9
DPI = 150

# ------------------------ NUMERIC HELPERS --------------------- #
def lap(A):
    return (np.roll(A,1,0)+np.roll(A,-1,0)+np.roll(A,1,1)+np.roll(A,-1,1)-4*A)

def grad(A):
    gx = 0.5*(np.roll(A,-1,0)-np.roll(A,1,0))
    gy = 0.5*(np.roll(A,-1,1)-np.roll(A,1,1))
    return gx, gy

def unit(vx, vy):
    m = np.sqrt(vx*vx+vy*vy) + EPS
    return vx/m, vy/m, m

def ema(prev, new, tau, dt):
    if tau <= 0: return new.copy()
    return prev + (dt/tau) * (new - prev)

def halfmax_Q(f, P):
    if len(f) < 4 or np.all(P <= 0): return 0.0, 0.0, 0.0
    i0 = int(np.argmax(P)); pf = float(f[i0]); pp = float(P[i0])
    if pp <= 0: return pf, 0.0, 0.0
    half = pp * 0.5
    iL = i0
    while iL > 0 and P[iL] > half: iL -= 1
    iR = i0
    while iR < len(P)-1 and P[iR] > half: iR += 1
    if iR <= iL: return pf, 0.0, 0.0
    bw = float(f[iR] - f[iL])
    Q = pf / bw if bw > 0 else 0.0
    return pf, bw, Q

def stable_clip(a, lim=1e6):
    return np.clip(a, -lim, lim)

def gaussian_drive(nx, ny, sigma_pix):
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = nx//2, ny//2
    R2 = (X-cx)**2 + (Y-cy)**2
    G = np.exp(-R2/(2.0*sigma_pix**2))
    G /= (G.sum() + EPS)       # normalize so drive ampl is in DRIVE_AMP
    return G

def anisotropy_ratio(y, Mx, My):
    """
    A = mean((M·∇y)^2), B = mean((M⊥·∇y)^2); ratio = A/B.
    >1 indicates preferred response along M.
    """
    gx, gy = grad(y)
    along = Mx*gx + My*gy
    perp  = -My*gx + Mx*gy
    A = float(np.mean(along*along))
    B = float(np.mean(perp*perp) + EPS)
    return A / B

# -------------------------- SIM ------------------------------- #
DRIVE_MASK = gaussian_drive(NX, NY, DRIVE_SIGMA)

def run_for_tau(tau_m):
    rng = np.random.default_rng(SEED)
    # State fields
    y  = 1e-3 * rng.standard_normal((NX, NY))  # displacement
    v  = np.zeros_like(y)                       # velocity
    ym = np.zeros_like(y)                       # memory EMA
    Mx = np.zeros_like(y)                       # magnet
    My = np.zeros_like(y)

    series = []  # spatial mean of |y| (avoid nodes)
    two_z_om0 = 2.0 * ZETA * OMEGA0

    for t in range(STEPS):
        # Drive
        if t < DRIVE_STEPS:
            y += DRIVE_AMP * np.sin(DRIVE_OMEGA * t * DT) * DRIVE_MASK

        # Spatial ops
        lap_y = lap(y)
        gx, gy = grad(y)
        nxg, nyg, gmag = unit(gx, gy)

        # Magnet update
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

        # Anisotropic advection
        adv = ALPHA * (Mx*gx + My*gy)

        # Second-order wave eqn with memory
        v += DT * (C*C*lap_y - two_z_om0*v - OMEGA0*OMEGA0*y + BETA*ym + adv)
        y += DT * v

        # Hygiene
        y *= 0.999
        y  = np.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        v  = np.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)

        if (t % SAVE_EVERY) == 0:
            series.append(float(np.mean(np.abs(y))))

    return np.asarray(series, float), Mx, My, y

# ---------------------- RUN, SAVE, ASSEMBLE ------------------- #
def main():
    rows = []
    spectra = []   # list of (tau, f, Pxx) for figure assembly
    last_snap = {} # one snapshot (Mx,My,y) for the largest tau

    for tau in TAU_M_LIST:
        series, Mx, My, y = run_for_tau(tau)

        # Welch spectrum on demeaned series
        fs = 1.0/(DT*SAVE_EVERY)
        nperseg = min(WELCH_NPERSEG, max(128, len(series)//2))
        f, Pxx = welch(series - series.mean(), fs=fs, nperseg=nperseg)
        pf, bw, Q = halfmax_Q(f, Pxx)
        ani = anisotropy_ratio(y, Mx, My)

        # Save per-tau plots
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

        rows.append(dict(
            tauM=tau, samples=len(series), fs=fs,
            peak_freq=pf, bandwidth_halfmax=bw, Q_factor=Q,
            anisotropy=ani
        ))
        spectra.append((tau, f, Pxx))
        last_snap = {"tau": tau, "Mx": Mx, "My": My, "y": y}

    # CSV summary
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "magnet_echo_summary.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote summary → {csv_path}")

    # === Assemble a one-page A4 PDF figure (Lock → Verify → Style) === #
    pdf_path = os.path.join(OUTDIR, "magnet_echo_figure.pdf")
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=200)  # A4 portrait
        gs = fig.add_gridspec(4, 2, height_ratios=[1,1,1,1], hspace=0.9, wspace=0.35)

        # (Top row) Spectra grid: first 3 taus
        for i, (tau, f, Pxx) in enumerate(spectra[:3]):
            ax = fig.add_subplot(gs[0, i if i < 2 else 1])  # place first 2; if 3rd, overwrite col 2
            ax.semilogy(f, Pxx + 1e-16)
            ax.set_title(f"Spectrum τₘ={tau:g}")
            ax.set_xlabel("f"); ax.set_ylabel("Power")
            ax.grid(True, which="both", alpha=0.3)

        # (2nd row) Spectra grid: next 3 taus (if present)
        row2 = spectra[3:6]
        for i, (tau, f, Pxx) in enumerate(row2[:2]):  # show up to 2 to keep layout clean
            ax = fig.add_subplot(gs[1, i])
            ax.semilogy(f, Pxx + 1e-16)
            ax.set_title(f"Spectrum τₘ={tau:g}")
            ax.set_xlabel("f"); ax.set_ylabel("Power")
            ax.grid(True, which="both", alpha=0.3)

        # (3rd row, left) Magnet quiver for largest tau
        if last_snap:
            tau = last_snap["tau"]; Mx, My, y = last_snap["Mx"], last_snap["My"], last_snap["y"]
            ax = fig.add_subplot(gs[2, 0])
            skip = 5
            U = Mx[::skip,::skip]; V = My[::skip,::skip]
            X, Y = np.meshgrid(np.arange(0,NX,skip), np.arange(0,NY,skip), indexing="ij")
            ax.quiver(Y, X, V, U, scale=60)
            ax.set_title(f"Magnet Field (τₘ={tau:g})")
            ax.set_xticks([]); ax.set_yticks([])
            ax.invert_yaxis()

            # (3rd row, right) Echo field snapshot
            ax2 = fig.add_subplot(gs[2, 1])
            im = ax2.imshow(y.T, origin="lower", interpolation="nearest")
            ax2.set_title(f"Echo Field |y| (τₘ={tau:g})")
            ax2.set_xticks([]); ax2.set_yticks([])
            cb = fig.colorbar(im, ax=ax2, shrink=0.8)

        # (Bottom row) Q vs τ and anisotropy vs τ
        axQ = fig.add_subplot(gs[3, 0])
        axA = fig.add_subplot(gs[3, 1])
        axQ.plot(df["tauM"], df["Q_factor"], marker="o")
        axQ.set_xlabel("τₘ"); axQ.set_ylabel("Q factor"); axQ.set_title("Q vs τₘ"); axQ.grid(True, alpha=0.3)
        axA.plot(df["tauM"], df["anisotropy"], marker="o")
        axA.set_xlabel("τₘ"); axA.set_ylabel("Anisotropy (‖ / ⟂)"); axA.set_title("Anisotropy vs τₘ"); axA.grid(True, alpha=0.3)

        fig.suptitle("Echo + Magnet: Spectra, Fields, and Scaling", y=0.995, fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Figure PDF → {pdf_path}")
    print(f"Figures & CSV in → {OUTDIR}/")
    # ---------------------- end main ---------------------- #

if __name__ == "__main__":
    main()
