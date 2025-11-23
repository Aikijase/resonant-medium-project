#!/usr/bin/env python3
"""
tools/magnet_echo_v9.py

Echo + Magnet — monotonic Q vs tauM with Lorentzian spectral fit
- Wave (y,v) + exponential memory ym with phase-lag force: + BETA_eff(tauM)*(ym - y)
- Damping scales down with tauM: zeta_eff(tauM)
- Auto-sweep drive ω per tauM; drive amplitude tapers (clamped) to keep linearity
- Dual Q metrics:
    * Q_spectrum: Lorentzian fit around the Welch peak (fallback: half-max)
    * Q_ringdown: exponential envelope during ring-down
- Anisotropy: alignment ⟨cos²θ⟩ ∈ [0,1] on top-|M| mask (15%)
- One-page A4 PDF figure + per-τ assets
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch, find_peaks

# --------------------- OUTPUTS --------------------- #
OUTDIR = "outputs/phaseXX"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------- GRID/TIME ------------------- #
NX, NY   = 96, 96
DT       = 0.01
DRIVE_STEPS   = 3500
RING_STEPS    = 4500
SAVE_EVERY    = 20
SEED     = 7
FS = 1.0 / (DT * SAVE_EVERY)
DPI = 150
EPS = 1e-9

# --------------------- PHYSICS --------------------- #
C       = 1.0
OMEGA0  = 1.50

# Memory scaling (steeper than prior)
BETA0   = 0.14
TAU_REF = 5.0
P_MEM   = 1.5  # stronger slope

# Damping scales down with tauM (less viscous as memory grows)
ZETA0   = 0.012
R_DAMP  = 0.60

# Magnet
SIGMA   = 0.85
GAMMA_M = 0.06
D_M     = 0.010
ALPHA   = 0.12
DAMP_M  = 0.0
M_SOFTCLIP = 200.0

# Tau sweep
TAU_LIST = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]

# Drive (directional plane wave, Gaussian window in y)
KX        = 2*np.pi/24.0
SIGMA_Y   = 14.0
DRIVE_AMP0= 0.026
AMP_MIN   = 0.012  # don’t let amplitude vanish at large tau

def drive_amp_for_tau(tau):
    amp = DRIVE_AMP0 * (tau/TAU_REF)**(-0.6)
    return max(AMP_MIN, float(amp))

def sweep_offsets_for_tau(tau):
    span = 0.22 * (TAU_REF/tau)**0.35
    return np.linspace(-span, +span, 17)

# --------------------- HELPERS --------------------- #
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
    return prev + (dt/tau)*(new - prev)

def halfmax_Q(f, P):
    if len(f) < 4 or np.all(P <= 0): return 0.0, 0.0, 0.0
    i0 = int(np.argmax(P)); pf = float(f[i0]); pp = float(P[i0])
    if pp <= 0: return pf, 0.0, 0.0
    half = pp*0.5
    iL = i0;  iR = i0
    while iL > 0 and P[iL] > half: iL -= 1
    while iR < len(P)-1 and P[iR] > half: iR += 1
    if iR <= iL: return pf, 0.0, 0.0
    bw = float(f[iR] - f[iL])
    Q  = pf/bw if bw>0 else 0.0
    return pf, bw, Q

def softclip_norm(Mx, My, mmax):
    mag = np.sqrt(Mx*Mx + My*My) + EPS
    s = np.minimum(1.0, mmax/mag)
    return Mx*s, My*s

def build_drive(nx, ny, kx, sigma_y):
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cy = ny//2
    gy = np.exp(-((Y - cy)**2)/(2.0*sigma_y**2))
    plane = np.cos(kx * X)
    m = np.mean(np.abs(plane*gy)) + EPS
    return (plane*gy)/m

DRIVE_MASK = build_drive(NX, NY, KX, SIGMA_Y)

def zeta_eff(tau):
    return ZETA0 * (tau/TAU_REF)**(-R_DAMP)

def beta_eff(tau):
    return BETA0 * (tau/TAU_REF)**(P_MEM)

def alignment_cos2(y, Mx, My, top_frac=0.15):
    gx, gy = grad(y)
    magM = np.sqrt(Mx*Mx + My*My)
    thresh = np.quantile(magM, 1.0 - top_frac)
    mask = magM >= thresh
    if not np.any(mask): return 0.5
    Mgx = (Mx*gx + My*gy)
    denom = np.sqrt((Mx*Mx+My*My)*(gx*gx+gy*gy)) + EPS
    cos2 = (Mgx/denom)**2
    return float(np.mean(cos2[mask]))

# ---------------- Lorentzian peak fit for Qspec --------------- #
def lorentz(x, A, f0, gamma, C):
    return A / (1.0 + ((x - f0)/gamma)**2) + C

def fit_lorentzian_Q(f, P):
    """Fit small window around the main peak; return (f0, gamma, Qspec)."""
    if len(f) < 10: return None
    i0 = int(np.argmax(P))
    # window of ~±10 bins around peak (clip to array)
    w = 10
    iL = max(0, i0 - w); iR = min(len(f)-1, i0 + w)
    fx = f[iL:iR+1]; Px = P[iL:iR+1]
    if len(fx) < 5: return None
    # initial guesses
    f0 = f[i0]
    A0 = float(P[i0] - np.median(Px))
    A0 = max(A0, np.max(Px)*0.1)
    C0 = float(np.median(Px))
    # estimate gamma from half-max width if possible
    pf, bw, _ = halfmax_Q(f, P)
    g0 = max(bw/2.0, (fx[1]-fx[0])*1.5)
    # simple least squares with bounds
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(lorentz, fx, Px, p0=[A0, f0, g0, C0],
                            bounds=([0.0, fx.min(), (fx[1]-fx[0])*0.5, 0.0],
                                    [np.inf, fx.max(), (fx[-1]-fx[0]), np.inf]),
                            maxfev=10000)
        A, f0, g, C = popt
        Qspec = f0/(2.0*g) if g>0 else 0.0
        return float(f0), float(g), float(Qspec)
    except Exception:
        return None

# ---------------- Simulation core ---------------- #
def simulate_one(tau, wdrive, amp, record_series=True):
    rng = np.random.default_rng(SEED)
    y  = 1e-3 * rng.standard_normal((NX, NY))
    v  = np.zeros_like(y)
    ym = np.zeros_like(y)
    Mx = np.zeros_like(y); My = np.zeros_like(y)
    zeta = zeta_eff(tau)
    beta = beta_eff(tau)
    two_z_om0 = 2.0 * zeta * OMEGA0

    s_drive, s_ring = [], []

    # DRIVE
    for t in range(DRIVE_STEPS):
        y += amp * np.sin(wdrive * t * DT) * DRIVE_MASK
        lap_y = lap(y)
        gx, gy = grad(y); nxg, nyg, gmag = unit(gx, gy)

        Mx += DT * (-Mx/max(tau,DT) + D_M*lap(Mx) + SIGMA*(gmag*nxg)
                    - GAMMA_M*(Mx*Mx+My*My)*Mx - DAMP_M*Mx)
        My += DT * (-My/max(tau,DT) + D_M*lap(My) + SIGMA*(gmag*nyg)
                    - GAMMA_M*(Mx*Mx+My*My)*My - DAMP_M*My)
        Mx, My = softclip_norm(Mx, My, M_SOFTCLIP)

        ym = ema(ym, y, tau, DT)
        mem = beta * (ym - y)
        adv = ALPHA * (Mx*gx + My*gy)

        v += DT * (C*C*lap_y - two_z_om0*v - OMEGA0*OMEGA0*y + mem + adv)
        y += DT * v
        y *= 0.999

        if record_series and (t % SAVE_EVERY)==0:
            s_drive.append(float(np.mean(np.abs(y))))

    # RING-DOWN
    for t in range(RING_STEPS):
        lap_y = lap(y)
        gx, gy = grad(y); nxg, nyg, gmag = unit(gx, gy)

        Mx += DT * (-Mx/max(tau,DT) + D_M*lap(Mx) + SIGMA*(gmag*nxg)
                    - GAMMA_M*(Mx*Mx+My*My)*Mx - DAMP_M*Mx)
        My += DT * (-My/max(tau,DT) + D_M*lap(My) + SIGMA*(gmag*nyg)
                    - GAMMA_M*(Mx*Mx+My*My)*My - DAMP_M*My)
        Mx, My = softclip_norm(Mx, My, M_SOFTCLIP)

        ym = ema(ym, y, tau, DT)
        mem = beta * (ym - y)
        adv = ALPHA * (Mx*gx + My*gy)

        v += DT * (C*C*lap_y - two_z_om0*v - OMEGA0*OMEGA0*y + mem + adv)
        y += DT * v
        y *= 0.999

        if record_series and (t % SAVE_EVERY)==0:
            s_ring.append(float(np.mean(np.abs(y))))

    return (np.asarray(s_drive), np.asarray(s_ring)), (Mx, My, y), zeta, beta

# ---------------- Run all taus ---------------- #
def spectral_metrics(series):
    f, Pxx = welch(series - series.mean(), fs=FS, nperseg=min(512, max(128, len(series)//2)))
    pf, bw, _Qhm = halfmax_Q(f, Pxx)
    fit = fit_lorentzian_Q(f, Pxx)
    if fit is not None:
        f0, gamma, Qspec = fit
    else:
        # Fallback to half-max estimate
        f0, gamma, Qspec = pf, bw/2.0 if bw>0 else 0.0, pf/bw if bw>0 else 0.0
    return f, Pxx, f0, gamma, Qspec

def ringdown_Q(series):
    if len(series) < 10: return 0.0
    s = np.abs(series - np.mean(series))
    peaks, _ = find_peaks(s)
    if len(peaks) < 6: return 0.0
    yv = np.log(s[peaks] + EPS); xv = peaks / FS
    A = np.vstack([xv, np.ones_like(xv)]).T
    k, _ = np.linalg.lstsq(A, yv, rcond=None)[0]
    if k >= 0: return 0.0
    f0 = 1.0 / np.mean(np.diff(peaks)/FS)
    return float(np.pi * f0 / abs(k)) if f0 > 0 else 0.0

def main():
    rows, spectra = [], []
    last_snap = {}

    for tau in TAU_LIST:
        amp = drive_amp_for_tau(tau)
        # pick best drive frequency by small sweep
        best = None
        for d in sweep_offsets_for_tau(tau):
            w = OMEGA0 * (1.0 + d)
            (sd,_),_,_,_ = simulate_one(tau, w, amp, record_series=True)
            f, Pxx, f0, gamma, Qs = spectral_metrics(sd)
            score = Pxx.max() if len(Pxx) else 0.0
            if (best is None) or (score > best[0]):
                best = (score, w, sd, f, Pxx, f0, gamma, Qs)
        score, w, sd, f, Pxx, f0, gamma, Qs = best

        # final run to grab fields and ring-down
        (sd, sr), (Mx, My, y), zeta, beta = simulate_one(tau, w, amp, record_series=True)
        f, Pxx, f0, gamma, Qs = spectral_metrics(sd)
        Qr = ringdown_Q(sr)
        align = alignment_cos2(y, Mx, My, top_frac=0.15)
        meanM = float(np.mean(np.sqrt(Mx*Mx + My*My)))

        # per-τ figs
        fig, ax = plt.subplots(figsize=(5,4), dpi=DPI)
        ax.semilogy(f, Pxx + 1e-16); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Spectrum τM={tau:g} (ω*={w:.3f})"); ax.set_xlabel("f"); ax.set_ylabel("Power")
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"echo_spectrum_tauM{int(tau)}.png")); plt.close(fig)

        tr = np.arange(len(sr))/FS
        fig, ax = plt.subplots(figsize=(5,3), dpi=DPI)
        ax.plot(tr, sr); ax.grid(True, alpha=0.3)
        ax.set_title(f"Ring-down τM={tau:g}"); ax.set_xlabel("time"); ax.set_ylabel("|y| mean")
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"ringdown_series_tauM{int(tau)}.png")); plt.close(fig)

        skip=5; X, Yg = np.meshgrid(np.arange(0,NX,skip), np.arange(0,NY,skip), indexing="ij")
        U, V = Mx[::skip,::skip], My[::skip,::skip]
        fig, ax = plt.subplots(figsize=(5,5), dpi=DPI)
        ax.quiver(Yg, X, V, U, scale=60); ax.invert_yaxis()
        ax.set_title(f"Magnet Field τM={tau:g}"); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"magnet_field_tauM{int(tau)}.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5,5), dpi=DPI)
        im = ax.imshow(y.T, origin="lower", interpolation="nearest")
        ax.set_title(f"Echo Field |y| τM={tau:g}"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"echo_field_tauM{int(tau)}.png")); plt.close(fig)

        rows.append(dict(
            tauM=tau,
            drive_omega=w,
            zeta_eff=zeta,
            BETA_eff=beta,
            f0_lorentz=f0,
            gamma_lorentz=gamma,
            Q_spectrum=Qs,
            Q_ringdown=Qr,
            alignment_cos2=align,
            mean_M=meanM
        ))
        spectra.append((tau, f, Pxx))
        last_snap = {"tau": tau, "Mx": Mx, "My": My, "y": y}

    # CSV
    df = pd.DataFrame(rows)
    csv = os.path.join(OUTDIR, "magnet_echo_summary.csv")
    df.to_csv(csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote summary → {csv}")

    # One-page A4 PDF
    pdf = os.path.join(OUTDIR, "magnet_echo_figure.pdf")
    with PdfPages(pdf) as pp:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
        gs = fig.add_gridspec(4, 2, height_ratios=[1,1,1,1], hspace=0.9, wspace=0.35)
        # spectra grids
        for i, (tau, f, Pxx) in enumerate(spectra[:2]):
            ax = fig.add_subplot(gs[0, i]); ax.semilogy(f, Pxx + 1e-16)
            ax.grid(True, which="both", alpha=0.3); ax.set_title(f"Spectrum τM={tau:g}")
            ax.set_xlabel("f"); ax.set_ylabel("Power")
        for i, (tau, f, Pxx) in enumerate(spectra[2:4]):
            ax = fig.add_subplot(gs[1, i]); ax.semilogy(f, Pxx + 1e-16)
            ax.grid(True, which="both", alpha=0.3); ax.set_title(f"Spectrum τM={tau:g}")
            ax.set_xlabel("f"); ax.set_ylabel("Power")
        # magnet & field (largest tau)
        if last_snap:
            tau = last_snap["tau"]; Mx, My, y = last_snap["Mx"], last_snap["My"], last_snap["y"]
            ax = fig.add_subplot(gs[2, 0]); skip = 5
            U, V = Mx[::skip,::skip], My[::skip,::skip]
            X, Yg = np.meshgrid(np.arange(0,NX,skip), np.arange(0,NY,skip), indexing="ij")
            ax.quiver(Yg, X, V, U, scale=60); ax.invert_yaxis()
            ax.set_title(f"Magnet Field τM={tau:g}"); ax.set_xticks([]); ax.set_yticks([])
            ax2 = fig.add_subplot(gs[2, 1])
            im = ax2.imshow(y.T, origin="lower", interpolation="nearest")
            ax2.set_title(f"Echo Field |y| τM={tau:g}"); ax2.set_xticks([]); ax2.set_yticks([])
            fig.colorbar(im, ax=ax2, shrink=0.8)
        # scaling
        axQ = fig.add_subplot(gs[3, 0]); axA = fig.add_subplot(gs[3, 1])
        axQ.plot(df["tauM"], df["Q_spectrum"], marker="o", label="spectral (Lorentz)")
        axQ.plot(df["tauM"], df["Q_ringdown"], marker="s", label="ring-down")
        axQ.set_xlabel("τM"); axQ.set_ylabel("Q"); axQ.set_title("Q vs τM")
        axQ.grid(True, alpha=0.3); axQ.legend()
        axA.plot(df["tauM"], df["alignment_cos2"], marker="o")
        axA.set_xlabel("τM"); axA.set_ylabel("Alignment ⟨cos²θ⟩")
        axA.set_title("Directional alignment (top |M|, 15%)"); axA.grid(True, alpha=0.3)
        fig.suptitle("Echo + Magnet v9: Monotonic Q with Memory-Scaled Damping", y=0.995, fontsize=12)
        pp.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"Figure PDF → {pdf}")
    print(f"Figures & CSV in → {OUTDIR}/")

if __name__ == "__main__":
    main()
