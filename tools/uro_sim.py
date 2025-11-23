#!/usr/bin/env python3
"""
URO Simulation Tool (Phase 9)
=============================

Single-file, drop-in simulator for the Unified Resonant Operator (URO).

Fixes in this version
---------------------
- **Argparse SystemExit:2**: `--out-prefix` is now **optional** with a safe default
  (`outputs/uro/<domain>_<YYYYMMDD-HHMMSS>`). You can still pass it explicitly.
- **Robust directories**: handles empty directory parts in `out-prefix`.
- **Manifest domain bug**: no longer relies on a global `args`.
- **Short runs safety**: guards for very small `T/dt`.
- **Built-in smoke tests**: `smoketest` subcommand to verify end-to-end outputs.

Core model (continuous form):
    A¨ + κ A˙ + Ω^2 A = F(t) + τ ∫ K(s) A(t-s) ds + ε g(A,t)

Discrete update (implemented here):
    M_{t+1}   = (1 - dt/Θ_eff) M_t + (dt/Θ_eff) A_t
    A˙_{t+1} = A˙_t + dt * [ F_t + τ*M_t + ε*g(A_t,t) - κ*A˙_t - Ω^2*A_t ]
    A_{t+1}   = A_t + dt * A˙_{t+1}

Features
--------
- Kernels: 1-pole or 2-pole exponential memory (fast/slow with mix w).
- Drives: sine, metronome (pulse train), step, noise, or CSV time series.
- Nonlinearities: none, cubic (A^3), gain (A*(1-αA^2)).
- Presets for three domains: cosmos, cognition, identity.
- Outputs: CSV (t, A, Adot, M, F), spectrum CSV (|FFT|), and PNG plots.
- Resonance diagnostics: peak frequency, Q-like bandwidth estimate.

Usage (now tolerant; `--out-prefix` optional)
--------------------------------------------
    python tools/uro_sim.py run --domain cognition --T 60 --dt 0.01 \
        --drive sine --drive-freq 1.0 --drive-amp 1.0 \
        --out-prefix outputs/uro/cog_demo

    # Defaults out-prefix to outputs/uro/cognition_<timestamp>
    python tools/uro_sim.py preset --domain cognition

    # Metronome pulses (e.g., 120 bpm = 2 Hz)
    python tools/uro_sim.py run --domain cognition --drive metronome --bpm 120 \
        --T 30 --dt 0.005 --out-prefix outputs/uro/cog_bpm120

    # Quick end-to-end checks
    python tools/uro_sim.py smoketest

Dependencies
------------
- numpy, pandas, matplotlib (no seaborn). Install in your venv.

"""
from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Core data structures
# -----------------------------

@dataclass
class UROParams:
    kappa: float     # damping κ
    Omega: float     # natural frequency Ω (rad/s in cognition; 1/e-fold in cosmos)
    tau: float       # memory gain τ
    Theta_f: float   # fast memory timescale Θ_f
    Theta_s: float   # slow  memory timescale Θ_s (may equal Θ_f for 1-pole)
    w: float         # mixture weight for fast pole (0..1); w=1 → 1-pole fast-only
    eps: float       # nonlinearity scaling ε
    alpha: float     # nonlinearity shape (used in gain)
    nonlin: str      # 'none'|'cubic'|'gain'

@dataclass
class SimConfig:
    domain: str      # 'cosmos'|'cognition'|'identity'
    T: float         # total simulated time
    dt: float        # time step
    drive: str       # 'sine'|'metronome'|'step'|'noise'|'csv'
    drive_amp: float # amplitude for sine/step/noise
    drive_freq: float # freq (Hz for cognition/identity; 1/e-fold for cosmos)
    bpm: float       # for metronome (beats per minute)
    csv_path: str    # optional external drive CSV with columns t,F
    seed: int        # RNG seed for noise drive

# -----------------------------
# Presets per domain (rough defaults; tune per dataset)
# -----------------------------

PRESETS: Dict[str, UROParams] = {
    # Cosmos works in log-a units, but simulator itself is unit-agnostic.
    'cosmos':    UROParams(kappa=0.15, Omega=0.50, tau=0.10, Theta_f=1.5, Theta_s=4.0, w=0.6, eps=0.02, alpha=0.3, nonlin='cubic'),
    'cognition': UROParams(kappa=0.80, Omega=1.00, tau=0.50, Theta_f=0.8, Theta_s=3.0, w=0.7, eps=0.05, alpha=0.2, nonlin='gain'),
    'identity':  UROParams(kappa=0.35, Omega=0.20, tau=0.30, Theta_f=6.0, Theta_s=48.0, w=0.5, eps=0.03, alpha=0.2, nonlin='cubic'),
}

# -----------------------------
# Kernels and nonlinearities
# -----------------------------

def update_memory(Mf: float, Ms: float, A: float, dt: float, Theta_f: float, Theta_s: float) -> Tuple[float, float]:
    """Update fast/slow exponential traces."""
    Theta_f = max(Theta_f, 1e-12)
    Theta_s = max(Theta_s, 1e-12)
    Mf = (1 - dt/Theta_f)*Mf + (dt/Theta_f)*A
    Ms = (1 - dt/Theta_s)*Ms + (dt/Theta_s)*A
    return Mf, Ms

def mix_memory(Mf: float, Ms: float, w: float) -> float:
    return w * Mf + (1 - w) * Ms

def nonlinearity(A: float, t: float, kind: str, alpha: float) -> float:
    if kind == 'none':
        return 0.0
    if kind == 'cubic':
        return A**3
    if kind == 'gain':
        # soft gain control: A*(1 - α A^2) — saturates for large |A|
        return A * (1 - alpha * (A**2))
    raise ValueError(f"Unknown nonlinearity: {kind}")

# -----------------------------
# Drives
# -----------------------------

def make_drive(cfg: SimConfig) -> Optional[Callable[[float], float]]:
    if cfg.drive == 'sine':
        w = 2*np.pi*cfg.drive_freq
        return lambda t: cfg.drive_amp * np.sin(w*t)
    if cfg.drive == 'metronome':
        freq = cfg.bpm / 60.0
        period = 1.0 / max(freq, 1e-9)
        width = 0.05 * period  # 5% duty pulse
        amp = cfg.drive_amp
        def F(t: float) -> float:
            phase = t % period
            return amp if phase < width else 0.0
        return F
    if cfg.drive == 'step':
        return lambda t: cfg.drive_amp
    if cfg.drive == 'noise':
        # special-cased in simulate
        return None
    if cfg.drive == 'csv':
        if not cfg.csv_path:
            raise FileNotFoundError("Drive CSV path not provided; use --csv-path <file>")
        if not os.path.exists(cfg.csv_path):
            raise FileNotFoundError(f"Drive CSV not found: {cfg.csv_path}")
        df = pd.read_csv(cfg.csv_path)
        if not {'t','F'}.issubset(df.columns):
            raise ValueError("Drive CSV must have columns t,F")
        t_vals = df['t'].to_numpy()
        f_vals = df['F'].to_numpy()
        def F(t: float) -> float:
            return np.interp(t, t_vals, f_vals, left=f_vals[0], right=f_vals[-1])
        return F
    raise ValueError(f"Unknown drive: {cfg.drive}")

# -----------------------------
# Simulator
# -----------------------------

def simulate(params: UROParams, cfg: SimConfig, out_prefix: str) -> Dict[str, float]:
    # Ensure output directory exists (support bare filenames) 
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Ensure enough steps for FFT/plots
    N = max(int(np.round(cfg.T / cfg.dt)), 4)
    t = np.linspace(0.0, cfg.T, N, endpoint=False)

    A = np.zeros(N)
    Adot = np.zeros(N)
    Mf = 0.0
    Ms = 0.0
    M = np.zeros(N)
    F = np.zeros(N)

    # Drive setup
    F_func = make_drive(cfg)
    rng = np.random.default_rng(cfg.seed)

    for i in range(N-1):
        ti = t[i]
        # Drive value
        if cfg.drive == 'noise':
            F[i] = cfg.drive_amp * rng.standard_normal()
        else:
            F[i] = float(F_func(ti)) if F_func is not None else 0.0
        # Memory updates
        Mf, Ms = update_memory(Mf, Ms, A[i], cfg.dt, params.Theta_f, params.Theta_s)
        M[i] = mix_memory(Mf, Ms, params.w)
        # Nonlinearity
        g = nonlinearity(A[i], ti, params.nonlin, params.alpha)
        # Acceleration update (Euler–Cromer style for stability on velocity)
        Adot[i+1] = Adot[i] + cfg.dt * ( F[i] + params.tau * M[i] + params.eps * g - params.kappa * Adot[i] - (params.Omega**2) * A[i] )
        A[i+1] = A[i] + cfg.dt * Adot[i+1]

    # Last F/M (safe even for small N because N>=4)
    F[-1] = F[-2]
    M[-1] = M[-2]

    # Save time series
    ts_path = f"{out_prefix}.timeseries.csv"
    df = pd.DataFrame({'t': t, 'A': A, 'Adot': Adot, 'M': M, 'F': F})
    df.to_csv(ts_path, index=False)

    # Spectrum and diagnostics
    window = np.hanning(N)
    A_win = A * window
    fft = np.fft.rfft(A_win)
    freqs = np.fft.rfftfreq(N, d=cfg.dt)
    amp = np.abs(fft)

    # Peak detection (ignore DC)
    if len(amp) > 1:
        idx_peak = 1 + int(np.argmax(amp[1:]))
        f_peak = float(freqs[idx_peak])
        a_peak = float(amp[idx_peak])
        half = a_peak / np.sqrt(2)
        left = idx_peak
        while left > 1 and amp[left] > half:
            left -= 1
        right = idx_peak
        while right < len(amp)-1 and amp[right] > half:
            right += 1
        bw = max(float(freqs[right] - freqs[left]), 1e-12)
        Q_est = f_peak / bw if bw > 0 else float('inf')
    else:
        f_peak, a_peak, Q_est = 0.0, 0.0, 0.0

    sp_path = f"{out_prefix}.spectrum.csv"
    pd.DataFrame({'f': freqs, 'amp': amp}).to_csv(sp_path, index=False)

    # Plots
    fig1 = plt.figure()
    plt.plot(t, A, label='A(t)')
    plt.plot(t, F, label='F(t)', alpha=0.6)
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.legend()
    plt.tight_layout()
    fig1.savefig(f"{out_prefix}.timeseries.png", dpi=160)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(freqs, amp, label='|FFT(A)|')
    if f_peak > 0:
        plt.axvline(f_peak, linestyle='--')
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.tight_layout()
    fig2.savefig(f"{out_prefix}.spectrum.png", dpi=160)
    plt.close(fig2)

    # Manifest
    manifest = {
        'domain': cfg.domain,
        'params': vars(params),
        'config': cfg.__dict__,
        'outputs': {
            'timeseries_csv': ts_path,
            'spectrum_csv': sp_path,
            'timeseries_png': f"{out_prefix}.timeseries.png",
            'spectrum_png': f"{out_prefix}.spectrum.png",
        },
        'diagnostics': {
            'f_peak': float(f_peak),
            'A_fft_peak': float(a_peak),
            'Q_est': float(Q_est),
        }
    }
    man_path = f"{out_prefix}.manifest.json"
    pd.Series(manifest).to_json(man_path)

    print("[uro] wrote:")
    for k,v in manifest['outputs'].items():
        print(f"  - {k}: {v}")
    print("[uro] diagnostics:")
    print(f"  f_peak={manifest['diagnostics']['f_peak']:.6g}  Q≈{manifest['diagnostics']['Q_est']:.3g}")

    return manifest

# -----------------------------
# CLI
# -----------------------------

def default_out_prefix(domain: str) -> str:
    ts = time.strftime('%Y%m%d-%H%M%S')
    return os.path.join('outputs', 'uro', f'{domain}_{ts}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='URO Simulation Tool (Phase 9)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # Shared defaults
    def add_common(run):
        run.add_argument('--domain', choices=list(PRESETS.keys()), default='cognition')
        run.add_argument('--kappa', type=float)
        run.add_argument('--Omega', type=float)
        run.add_argument('--tau', type=float)
        run.add_argument('--Theta-f', type=float, dest='Theta_f')
        run.add_argument('--Theta-s', type=float, dest='Theta_s')
        run.add_argument('--w', type=float)
        run.add_argument('--eps', type=float)
        run.add_argument('--alpha', type=float)
        run.add_argument('--nonlin', choices=['none','cubic','gain'])
        run.add_argument('--T', type=float, default=60.0, help='total time')
        run.add_argument('--dt', type=float, default=0.01, help='time step')
        run.add_argument('--drive', choices=['sine','metronome','step','noise','csv'], default='sine')
        run.add_argument('--drive-amp', type=float, default=1.0)
        run.add_argument('--drive-freq', type=float, default=1.0)
        run.add_argument('--bpm', type=float, default=120.0)
        run.add_argument('--csv-path', type=str, default='')
        run.add_argument('--seed', type=int, default=42)
        run.add_argument('--out-prefix', type=str, default=None, help='where to write outputs (default: outputs/uro/<domain>_<timestamp>)')
        return run

    add_common(sub.add_parser('run', help='Run a simulation with specified parameters'))
    add_common(sub.add_parser('preset', help='Run with domain preset values'))

    # Smoketest command (runs 3 tiny sims and checks files exist)
    st = sub.add_parser('smoketest', help='Run end-to-end smoke tests (outputs under outputs/uro/tests)')
    st.add_argument('--keep', action='store_true', help='keep test outputs (default keeps)')

    return p


def params_from_args(args: argparse.Namespace) -> UROParams:
    base = PRESETS[args.domain]
    def pick(name, default):
        val = getattr(args, name, None)
        return default if val is None else val
    return UROParams(
        kappa=pick('kappa', base.kappa),
        Omega=pick('Omega', base.Omega),
        tau=pick('tau', base.tau),
        Theta_f=pick('Theta_f', base.Theta_f),
        Theta_s=pick('Theta_s', base.Theta_s),
        w=pick('w', base.w),
        eps=pick('eps', base.eps),
        alpha=pick('alpha', base.alpha),
        nonlin=pick('nonlin', base.nonlin),
    )


def cfg_from_args(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        domain=args.domain,
        T=args.T, dt=args.dt, drive=args.drive, drive_amp=args.drive_amp,
        drive_freq=args.drive_freq, bpm=args.bpm, csv_path=args.csv_path,
        seed=args.seed
    )


def run_smoketests() -> int:
    """Run tiny simulations to verify that outputs are produced correctly."""
    print('[uro:test] starting smoke tests...')
    base = os.path.join('outputs', 'uro', 'tests')
    os.makedirs(base, exist_ok=True)

    cases = [
        ('cognition', 'sine',     dict(T=2.0, dt=0.005, drive_freq=1.0)),
        ('cognition', 'metronome',dict(T=2.0, dt=0.002, bpm=120.0)),
        ('identity',  'step',     dict(T=10.0, dt=0.1,  drive_amp=0.4)),
    ]

    ok = True
    for idx, (domain, drive, overrides) in enumerate(cases, 1):
        out = os.path.join(base, f'{domain}_{drive}_{idx}')
        # Build args-like container
        class A: pass
        a = A()
        a.domain = domain
        a.T = overrides.get('T', 2.0)
        a.dt = overrides.get('dt', 0.01)
        a.drive = drive
        a.drive_amp = overrides.get('drive_amp', 1.0)
        a.drive_freq = overrides.get('drive_freq', 1.0)
        a.bpm = overrides.get('bpm', 120.0)
        a.csv_path = ''
        a.seed = 123 + idx
        a.kappa=a.Omega=a.tau=a.Theta_f=a.Theta_s=a.w=a.eps=a.alpha=a.nonlin=None

        params = params_from_args(a)
        cfg = cfg_from_args(a)
        simulate(params, cfg, out)

        # Check files exist
        for suf in ['.timeseries.csv','.spectrum.csv','.timeseries.png','.spectrum.png','.manifest.json']:
            path = out + suf
            if not os.path.exists(path):
                print(f'[uro:test] MISSING {path}')
                ok = False

    print('[uro:test] all tests passed' if ok else '[uro:test] tests FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == 'smoketest':
        sys.exit(run_smoketests())

    params = params_from_args(args)
    cfg = cfg_from_args(args)

    # If subcommand is 'preset', we ignore manual overrides and use domain defaults
    if args.cmd == 'preset':
        params = PRESETS[args.domain]

    out_prefix = args.out_prefix or default_out_prefix(args.domain)
    simulate(params, cfg, out_prefix)
