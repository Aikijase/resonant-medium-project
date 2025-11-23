#!/usr/bin/env python3
"""
tools/magnet_echo_v4.py

Echo (finite-memory) + Magnet demo — robust anisotropy & sharper Q vs tau_M
- Second-order wave (y, v)
- Exponential memory ym with force term: + BETA * (ym - y)
  (phase-lag sharpening; larger tau_M narrows the resonance)
- Directional plane-wave drive to seed alignment
- Masked anisotropy uses median((M·∇y)^2 / ((M⊥·∇y)^2 + ridge)) on top-|M| region
- Saves spectra, quiver, field snapshot, CSV summary, and a one-page A4 PDF

Outputs → outputs/phaseXX/:
  magnet_echo_summary.csv
  echo_spectrum_tauM*.png
  magnet_field_tauM*.png
  echo_field_tauM*.png
  magnet_echo_figure.pdf
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch

# --------------------------- CONFIG --------------------------- #
OUTDIR = "outputs/phaseXX"; os.makedirs(OUTDIR, exist_ok=True)

# Grid / time
NX, NY     = 96, 96
DT         = 0.01
STEPS      = 9000
SAVE_EVERY = 20
SEED       = 7

# Wave + Echo params (tuned for Q growth with tau_M)
C        = 1.0
OMEGA0   = 1.50
ZETA     = 0.010          # light baseline damping
BETA     = 0.32           # memory force strength (acts on (ym - y))

# Memory sweep (viscosity proxy)
TAU_M_LIST = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]

# Magnet params (directional bias)
SIGMA    = 0.90           # source from |∇y|
GAMMA_M  = 0.06           # saturation
D_M      = 0.008          # diffusion (small: keep domains)
ALPHA    = 0.16           # anisotropic advection (M·∇)y
DAMP_M   = 0.000

# Directional plane-wave drive (x-directed), gaussian window in y
DRIVE_STEPS = 3500
DRIVE_OMEGA = 1.44        # close to OMEGA0
DRIVE_AMP   = 0.028
KX          = 2*np.pi/24.0
DRIVE_SIGMA = 14.0

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
    Q  = pf / bw if bw > 0 else 0.0
    return pf, bw, Q

def stable_clip(a, lim=1e6):
    return np.clip(a, -lim, lim)

def build_drive(nx, ny, kx, sigma_y):
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cy = ny//2
    gy = np.exp(-((Y - cy)**2)/(2.0*sigma_y**2))
    plane = np.cos(kx * X)
    m = np.mean(np.abs(plane*gy)) + EPS
    return (plane * gy) / m

DRIVE_MASK = build_drive(NX, NY, KX, DRIVE_SIGMA)

# -------- robust masked anisotropy: median ratio with ridge --- #
def anisotropy_ratio_masked(y, Mx, My, top_frac=0.30, clip=(0.1, 100.0)):
    gx, gy = grad(y)
    magM = np.sqrt(Mx*Mx + My*My)
    thresh = np.quantile(magM, 1.0 - top_frac)
    mask = magM >= thresh
    if not np.any(mask): return 1.0, 0.0  # neutral

    along = (Mx*gx + My*gy)[mask]
    perp  = (-My*gx + Mx*gy)[mask]
    A2 = along*along
    B2 = perp*perp

    # ridge based on local scale of B2 (avoid divide-by-near-zero)
    ridge = np.quantile(B2, 0.10) + EPS
    r = A2 / (B2 + ridge)
    med = float(np.median(r))
    med_clipped = float(np.clip(med, clip[0], clip[1]))
    return med, med_clipped

# -------------------------- SIM ------------------------------- #
def run_for_tau(tau_m):
    rng = np.random.default_rng(SEED)
    y  = 1e-3 * rng.standard_normal((NX, NY))
    v  = np.zeros_like(y)
    ym = np.zeros_like(y)
    Mx = np.zeros_like(y); My = np.zeros_like(y)

    series = []
    two_z_om0 = 2.0 * ZETA * OMEGA0

    for t in range(STEPS):
        if t < DRIVE_STEPS:
            y += DRIVE_AMP * np.sin(DRIVE_OMEGA * t * DT) * DRIVE_MASK

        lap_y = lap(y)
        gx, gy = grad(y)
        nxg, nyg, gmag = unit(gx, gy)

        # Magnet
        Mx += DT * (
            -Mx/max(tau_m, DT) + D_M*lap(Mx) + SIGMA*(gmag*nxg)
            - GAMMA_M*(Mx*Mx+My*My)*Mx - DAMP_M*Mx
        )
        My += DT * (
            -My/max(tau_m, DT) + D_M*lap(My) + SIGMA*(gmag*nyg)
            - GAMMA_M*(Mx*Mx+My*My)*My - DAMP_M*My
        )
        Mx = stable_clip(Mx, 1e3); My = stable_clip(My, 1e3)

        # Memory update and phase-lag force
        ym = ema(ym, y, tau=tau_m, dt=DT)
        mem_force = BETA * (ym - y)

        # Anisotropic advection
        adv = ALPHA * (Mx*gx + My*gy)

        # Wave + memory
        v += DT * (C*C*lap_y - two_z_om0*v - OMEGA0*OMEGA0*y + mem_force + adv)
        y += DT * v

        # hygiene
        y *= 0.999
        y = np.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        v = np.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)

        if (t % SAVE_EVERY) == 0:
            series.append(float(np.mean(np.abs(y))))

    return np.asarray(series, float), Mx, My, y

# ---------------------- RUN, SAVE, ASSEMBLE ------------------- #
def main():
    rows, spectra = [], []
    last_snap = {}

    for tau in TAU_M_LIST:
        series, Mx, My, y = run_for_tau(tau)

        # Spectrum
        fs = 1.0/(DT*SAVE_EVERY)
        nperseg = min(WELCH_NPERSEG, max(128, len(series)//2))
        f, Pxx = welch(series - series.mean(), fs=fs, nperseg=nperseg)
        pf, bw, Q = halfmax_Q(f, Pxx)

        # Masked anisotropy (report raw median and clipped)
        ani_raw, ani_clip = anisotropy_ratio_masked(y, Mx, My, top_frac=0.30)

        # Save per-tau plots
        fig, ax = plt.subplots(figsize=(5,4), dpi=DPI)
        ax.semilogy(f, Pxx + 1e-16)
        ax.set_title(f"Echo Spectrum (tauM={tau:g})")
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
        ax.set_title(f"Magnet Field Snapshot (tauM={tau:g})")
        ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"magnet_field_tauM{int(tau)}.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5,5), dpi=DPI)
        im = ax.imshow(y.T, origin="lower", interpolation="nearest")
        ax.set_title(f"Echo Field |y| Snapshot (tauM={tau:g})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"echo_field_tauM{int(tau)}.png"))
        plt.close(fig)

        rows.append(dict(
            tauM=tau, samples=len(series), fs=fs,
            peak_freq=pf, bandwidth_halfmax=bw, Q_factor=Q,
            anisotropy_masked_raw=ani_raw,
            anisotropy_masked_clipped=ani_clip,
            mean_M=np.mean(np.sqrt(Mx*Mx+My*My))
        ))
        spectra.append((tau, f, Pxx))
        last_snap = {"tau": tau, "Mx": Mx, "My": My, "y": y}

    # CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "magnet_echo_summary.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote summary → {csv_path}")

    # One-page A4 PDF
    pdf_path = os.path.join(OUTDIR, "magnet_echo_figure.pdf")
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
        gs = fig.add_gridspec(4, 2, height_ratios=[1,1,1,1], hspace=0.9, wspace=0.35)

        # Spectra rows
        for i, (tau, f, Pxx) in enumerate(spectra[:2]):
            ax = fig.add_subplot(gs[0, i])
            ax.semilogy(f, Pxx + 1e-16); ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"Spectrum tauM={tau:g}"); ax.set_xlabel("f"); ax.set_ylabel("Power")
        for i, (tau, f, Pxx) in enumerate(spectra[2:4]):
            ax = fig.add_subplot(gs[1, i])
            ax.semilogy(f, Pxx + 1e-16); ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"Spectrum tauM={tau:g}"); ax.set_xlabel("f"); ax.set_ylabel("Power")

        # Magnet & field (largest tau)
        if last_snap:
            tau = last_snap["tau"]; Mx, My, y = last_snap["Mx"], last_snap["My"], last_snap["y"]
            ax = fig.add_subplot(gs[2, 0])
            skip = 5
            U = Mx[::skip,::skip]; V = My[::skip,::skip]
            X, Y = np.meshgrid(np.arange(0,NX,skip), np.arange(0,NY,skip), indexing="ij")
            ax.quiver(Y, X, V, U, scale=60)
            ax.set_title(f"Magnet Field (tauM={tau:g})"); ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis()

            ax2 = fig.add_subplot(gs[2, 1])
            im = ax2.imshow(y.T, origin="lower", interpolation="nearest")
            ax2.set_title(f"Echo Field |y| (tauM={tau:g})"); ax2.set_xticks([]); ax2.set_yticks([])
            fig.colorbar(im, ax=ax2, shrink=0.8)

        # Scaling
        axQ = fig.add_subplot(gs[3, 0]); axA = fig.add_subplot(gs[3, 1])
        axQ.plot(df["tauM"], df["Q_factor"], marker="o"); axQ.grid(True, alpha=0.3)
        axQ.set_xlabel("tauM"); axQ.set_ylabel("Q factor"); axQ.set_title("Q vs tauM")

        axA.plot(df["tauM"], df["anisotropy_masked_clipped"], marker="o"); axA.grid(True, alpha=0.3)
        axA.set_xlabel("tauM"); axA.set_ylabel("Anisotropy (masked, clipped)"); axA.set_title("Anisotropy vs tauM")

        fig.suptitle("Echo + Magnet: Robust Anisotropy & Memory-Sharpened Spectra", y=0.995, fontsize=12)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"Figure PDF → {pdf_path}")
    print(f"Figures & CSV in → {OUTDIR}/")

if __name__ == "__main__":
    main()
