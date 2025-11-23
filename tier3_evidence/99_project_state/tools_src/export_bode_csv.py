#!/usr/bin/env python3
"""
Export Bode curves (omega, |R|^2, gain[dB], phase) using saved artifacts.

Input: <stem>_residual_series.csv, <stem>_summary.json
Output: <stem>_bode.csv   (omega, P, gain_db, phase)
        <stem>_bode.png   (quick visual)
"""
import json, sys, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def omega_axis(Lz: int, dx: float) -> np.ndarray:
    return 2.0 * np.pi * np.fft.rfftfreq(Lz, d=dx)

def rms_norm(w: np.ndarray) -> np.ndarray:
    return w / (np.sqrt(np.mean(w*w)) + 1e-30)

def main():
    if len(sys.argv) < 2:
        print("Usage: export_bode_csv.py <out_stem>"); sys.exit(1)
    stem = Path(sys.argv[1])

    J = json.load(open(f"{stem}_summary.json"))
    L   = int(J["L"]); Lz = int(J["Lz"]); pad = int(J["pad"])

    # load residuals (log-detrended diff in the analyzer)
    arr = np.loadtxt(f"{stem}_residual_series.csv", delimiter=",", skiprows=1)
    a = arr[:,0]; res = arr[:,1]
    x  = np.linspace(0.0, 1.0, len(a))
    dx = x[1] - x[0] if len(x) > 1 else 1.0

    # spectrum from residuals (should match analyzer)
    R = np.fft.rfft(np.pad(res, (0, pad)))
    P = np.abs(R)**2
    omega = omega_axis(Lz, dx)

    # We don't have fs8_l and fs8_m here, so we'll export |R|^2 and phase of R
    # (gain/phase between model/LCDM were exported as a plot in analyzer; for CSV,
    # the most portable piece is residual power & phase(R).)
    phase_R = np.angle(R)

    out_csv = f"{stem}_bode.csv"
    np.savetxt(out_csv, np.column_stack([omega, P, phase_R]),
               delimiter=",", header="omega,P_residual,phase_R", comments="")

    # quick figure
    fig, ax = plt.subplots(2,1, figsize=(8,6))
    ax[0].plot(omega, P); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("ω"); ax[0].set_ylabel("|R(ω)|²"); ax[0].set_title("Residual Spectrum")
    ax[1].plot(omega, phase_R); ax[1].set_xscale("log")
    ax[1].set_xlabel("ω"); ax[1].set_ylabel("arg R(ω) [rad]"); ax[1].set_title("Residual Phase")
    out_png = f"{stem}_bode.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)

    print(f"Wrote {out_csv} and {out_png}")

if __name__ == "__main__":
    main()
