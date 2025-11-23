#!/usr/bin/env python3
"""
Phase-8 Spectral Analyzer — clean, zero-padded, with CSV+SVG outputs.

Outputs (for --out-stem=<STEM>):
  <STEM>_residual_series.csv     # a, residual_logdiff
  <STEM>_bode.csv                # omega, P_residual, gain_db, phase
  <STEM>_summary.json            # sizes + peak info (+ lorentz if fit)
  <STEM>_spectrum.png/.svg       # spectrum + Bode plot

Optional:
  --quadratic-peak   (parabolic peak refine)
  --lorentzian-fit   (fit A / (1 + ((x-x0)/gamma)^2) + C)  → Q = x0/(2γ)
  --open             (try to open SVG in browser after save)

It will use tools/phase8_spectral/fs8_provider.py if present:
  from fs8_provider import Params, get_fs8_series
Otherwise it falls back to a small toy generator.
"""
import argparse, json, os, math, webbrowser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- optional provider hook ----------
USE_PROVIDER = False
try:
    from fs8_provider import Params, get_fs8_series  # type: ignore
    USE_PROVIDER = True
except Exception:
    class Params:  # fallback
        def __init__(self, mu0, nu0, tau, kappa, sigma8, Om):
            self.mu0, self.nu0, self.tau, self.kappa, self.sigma8, self.Om = \
                mu0, nu0, tau, kappa, sigma8, Om

def _toy_series(a: np.ndarray, p: Params):
    # smooth LCDM-ish baseline
    fs8_l = p.sigma8 * (a ** (0.55 * p.Om)) * (1 - 0.05*(1 - a))
    # tiny resonant tweak
    amp   = 0.08 * (p.mu0 - 0.9)
    phase = 8.0 * p.nu0
    damp  = np.exp(-p.tau * (1 - a))
    drift = 1.0 + p.kappa * (1 - a)
    osc   = amp * damp * np.sin(-phase * np.log(a + 1e-12))
    fs8_m = fs8_l * drift * (1.0 + osc)
    eps = 1e-12
    return np.clip(fs8_l, eps, None), np.clip(fs8_m, eps, None)

# ---------- utils ----------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def make_Lz(L: int, zero_pad: int) -> int:
    base = next_pow2(max(1, L))
    return int(base * (2 ** max(0, int(zero_pad))))

def rms_norm(w: np.ndarray) -> np.ndarray:
    return w / (np.sqrt(np.mean(w*w)) + 1e-30)

def omega_axis(Lz: int, dx: float) -> np.ndarray:
    return 2.0 * np.pi * np.fft.rfftfreq(Lz, d=dx)

def poly_detrend_log(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 1e-12, None)
    ly = np.log(y)
    p  = np.polyfit(x, ly, 1)
    return ly - np.polyval(p, x)

def try_lorentz_fit(omega: np.ndarray, P: np.ndarray, imax: int):
    """
    Robust Lorentzian fit around the true peak:
      - window by half-power around argmax
      - constrain baseline C near local floor
      - weight toward the peak to avoid floor bias
    Returns dict or None.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return None

    # ---- peak-centered window by half-power (or relaxed threshold if narrow) ----
    Pmax = float(P[imax])
    if not np.isfinite(Pmax) or Pmax <= 0:
        return None

    # Start with half-power; if too few points, relax to 0.3*Pmax
    for frac in (0.5, 0.4, 0.3):
        mask = P >= (frac * Pmax)
        if np.count_nonzero(mask) >= 9:
            break
    idx = np.where(mask)[0]
    lo = max(0, idx[0] - 5); hi = min(len(P), idx[-1] + 6)
    xs = omega[lo:hi]; ys = P[lo:hi]
    if xs.size < 9:
        # fallback: fixed span around peak
        span = 30
        lo = max(0, imax - span); hi = min(len(omega), imax + span + 1)
        xs = omega[lo:hi]; ys = P[lo:hi]
        if xs.size < 9:
            return None

    x0_guess = float(omega[imax])
    dω = float(np.median(np.diff(xs))) if xs.size > 1 else max(1e-6, float(omega[1]-omega[0]) if len(omega)>1 else 1e-3)

    # ---- baseline estimate from local window (use a low percentile) ----
    C0 = float(np.percentile(ys, 10))
    C0 = max(0.0, C0)
    A0 = float(max(ys.max() - C0, 1e-12))
    g0 = max(0.15 * x0_guess, dω)  # moderate width

    # Emphasize near-peak points (heavier weight near x0_guess)
    weights = 1.0 / (1.0 + ((xs - x0_guess) / (3*dω))**2)
    sigma = 1.0 / np.maximum(weights, 1e-6)

    def lorentz(u, A, x0, gamma, C):
        return A / (1.0 + ((u - x0)/gamma)**2) + C

    # ---- bounds to keep it sane and near the peak ----
    lower = [0.0, 0.8 * x0_guess, max(dω, 1e-12), 0.5 * C0]
    upper = [np.inf, 1.2 * x0_guess, 0.6 * x0_guess, 1.2 * max(C0, 1e-12)]
    p0    = [A0,  x0_guess, g0, C0]

    try:
        popt, _ = curve_fit(lorentz, xs, ys, p0=p0, bounds=(lower, upper),
                            sigma=sigma, absolute_sigma=False, maxfev=60000)
        A_hat, x0_hat, gamma_hat, C_hat = map(float, popt)
        Q = (x0_hat / (2.0*gamma_hat)) if gamma_hat > 0 else None
        return dict(A=A_hat, omega0=x0_hat, gamma=gamma_hat, C=C_hat, Q=Q)
    except Exception:
        return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu0", type=float, required=True)
    ap.add_argument("--nu0", type=float, required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--sigma8", type=float, required=True)
    ap.add_argument("--Om", type=float, required=True)
    ap.add_argument("--a-min", type=float, dest="a_min", default=1e-3)
    ap.add_argument("--a-max", type=float, dest="a_max", default=1.0)
    ap.add_argument("--n-steps", type=int, default=4096)
    ap.add_argument("--zero-pad", type=int, default=4)
    ap.add_argument("--quadratic-peak", action="store_true")
    ap.add_argument("--lorentzian-fit", action="store_true")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--out-stem", type=str, required=True)
    args = ap.parse_args()

    ensure_dir(args.out_stem + "_dummy")

    # domain
    a  = np.linspace(args.a_min, args.a_max, args.n_steps)
    x  = np.linspace(0.0, 1.0, args.n_steps)
    dx = x[1] - x[0] if len(x) > 1 else 1.0

    # series
    p = Params(args.mu0, args.nu0, args.tau, args.kappa, args.sigma8, args.Om)
    if USE_PROVIDER:
        fs8_l, fs8_m = get_fs8_series(a, p)  # from your provider
    else:
        fs8_l, fs8_m = _toy_series(a, p)     # fallback

    # window + residuals (log space)
    w = rms_norm(np.hanning(len(x)))
    res_l = poly_detrend_log(x, fs8_l)
    res_m = poly_detrend_log(x, fs8_m)
    res   = (res_m - res_l) * w

    # zero-padding
    L  = len(res)
    Lz = make_Lz(L, args.zero_pad)
    pad = max(0, Lz - L)

    # frequency axis
    omega = omega_axis(Lz, dx)

    # residual spectrum
    R = np.fft.rfft(np.pad(res, (0, pad)))
    P = np.abs(R)**2

    # bode-like gain/phase (match zero-padding!)
    eps = 1e-12
    xl, xm = np.clip(fs8_l, eps, None), np.clip(fs8_m, eps, None)
    tl = np.log(xl); tm = np.log(xm)
    pl = np.polyval(np.polyfit(x, tl, 1), x)
    pm = np.polyval(np.polyfit(x, tm, 1), x)
    s_l = (tl - pl) * w
    s_m = (tm - pm) * w
    Yl  = np.fft.rfft(np.pad(s_l, (0, pad)))
    Ym  = np.fft.rfft(np.pad(s_m, (0, pad)))
    cross = Ym * np.conj(Yl)
    gain  = (np.abs(Ym) + 1e-300) / (np.abs(Yl) + 1e-300)
    phase = np.unwrap(np.angle(cross))

    assert omega.size == gain.size == phase.size, \
        f"FFT axis mismatch: {omega.size} vs {gain.size} vs {phase.size}"

    # ---- peak finding ----
    peak = {}
    P_use = P.copy()
    if P_use.size > 1:
        P_use[0] = 0.0  # ignore DC
    imax = int(np.argmax(P_use))
    peak["omega_at_max"] = float(omega[imax])
    peak["peak_power"]   = float(P[imax])

    if args.quadratic_peak and 1 <= imax < (len(omega) - 1):
        xs = omega[imax-1:imax+2]
        ys = P[imax-1:imax+2]
        M  = np.vstack([xs**2, xs, np.ones_like(xs)]).T
        qa, qb, qc = np.linalg.lstsq(M, ys, rcond=None)[0]
        if qa != 0:
            xv = -qb / (2*qa); yv = qa*xv*xv + qb*xv + qc
            peak["quad_peak_omega"] = float(xv)
            peak["quad_peak_power"] = float(yv)

    if args.lorentzian_fit:
        Lfit = try_lorentz_fit(omega, P, imax)
        if Lfit is not None:
            peak["lorentzian"] = Lfit

    # ---- save residual series ----
    out_csv_series = f"{args.out_stem}_residual_series.csv"
    np.savetxt(out_csv_series, np.column_stack([a, res]),
               delimiter=",", header="a,residual_logdiff", comments="")

    # ---- plot + save images ----
    # downsample for plotting speed if huge
    plot_idx = np.arange(0, omega.size, max(1, omega.size // 4000))
    w_plot = omega[plot_idx]; P_plot = P[plot_idx]
    gain_plot = gain[plot_idx]; phase_plot = phase[plot_idx]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(w_plot, P_plot)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("ω"); ax[0].set_ylabel("|R(ω)|²")
    ax[0].set_title("Residual Spectrum")
    if "lorentzian" in peak:
        # overlay a fitted curve near peak for visual check
        Lfit = peak["lorentzian"]
        x0   = Lfit["omega0"]; g = Lfit["gamma"]; A = Lfit["A"]; C = Lfit["C"]
        xs = np.logspace(np.log10(max(w_plot[1], x0/5)), np.log10(min(w_plot[-1], x0*5)), 400)
        ys = A / (1.0 + ((xs - x0)/g)**2) + C
        ax[0].plot(xs, ys, linewidth=1)

    ax[1].plot(w_plot, 20*np.log10(gain_plot), label="Gain [dB]")
    ax[1].plot(w_plot, phase_plot, label="Phase [rad]")
    ax[1].set_xscale("log")
    ax[1].legend(loc="upper right", frameon=False)
    ax[1].set_xlabel("ω"); ax[1].set_title("Bode-like Gain/Phase")

    out_png = f"{args.out_stem}_spectrum.png"
    out_svg = f"{args.out_stem}_spectrum.svg"
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.savefig(out_svg); plt.close(fig)

    # ---- Save Bode CSV ----
    out_csv_bode = f"{args.out_stem}_bode.csv"
    gain_db = 20*np.log10(gain + 1e-300)
    np.savetxt(out_csv_bode,
               np.column_stack([omega, P, gain_db, phase]),
               delimiter=",", header="omega,P_residual,gain_db,phase", comments="")

    # ---- summary ----
    summary = dict(L=int(L), Lz=int(Lz), pad=int(pad))
    summary.update(peak)
    out_json = f"{args.out_stem}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[wrote] {out_csv_series}")
    print(f"[wrote] {out_csv_bode}")
    print(f"[wrote] {out_json}")
    print(f"[wrote] {out_png}")
    print(f"[wrote] {out_svg}")

    if args.open:
        try:
            webbrowser.open(Path(out_svg).resolve().as_uri())
        except Exception:
            pass

if __name__ == "__main__":
    main()
