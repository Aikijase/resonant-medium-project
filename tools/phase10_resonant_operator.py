"""
Phase 10 — Unified Resonant Operator (Minimal Working Script)
Drop-in location suggestion: tools/phase10_resonant_operator.py
Outputs: outputs/phase10/<prefix>_*.png and <prefix>_summary.json

Usage examples:
  python tools/phase10_resonant_operator.py --preset neuron --steps 4000 --dt 0.01 --prefix quick_neuron
  python tools/phase10_resonant_operator.py --memory 0.30 --softness 0.70 --omega 2.1 --gamma 0.12 --gain 0.65 --noise 0.01 --steps 5000 --prefix custom

Notes:
- No external deps beyond numpy/matplotlib.
- Single-file, no patching; can be moved under your repo's tools/ directory.
- Creates simple time-series, phase-space, and spectrum plots + a JSON summary.
"""

import os, json, argparse, numpy as np
import matplotlib.pyplot as plt

# --------- Core Operator ---------
class ResonantOperator:
    """
    A minimal universal resonant loop with memory and softness.
    State: position x, velocity v.
    Memory: exponential moving average m of x with timescale tau.
    Dynamics (semi-implicit Euler):
      a = -omega^2 * f_soft(x, softness) - gamma * v + gain * (m - x) + xi
      v <- v + a*dt
      x <- x + v*dt
      m <- m + dt/tau * (x - m)
    Where xi is Gaussian noise scaled by 'noise' and 'interference' if desired.
    """
    def __init__(self, memory=0.3, softness=0.7, omega=2.0, gamma=0.1, gain=0.6, noise=0.0, seed=42):
        self.tau = float(memory) if memory > 1e-6 else 1e-6
        self.soft = float(softness)
        self.omega = float(omega)
        self.gamma = float(gamma)
        self.gain = float(gain)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        # state
        self.x = 0.1
        self.v = 0.0
        self.m = 0.0

    def f_soft(self, x):
        # Soft nonlinearity: interpolate between linear (soft→1) and saturating tanh (soft→0)
        # softness in [0,1]; effective function = soft*x + (1-soft)*tanh(x)
        s = np.clip(self.soft, 0.0, 1.0)
        return s*x + (1.0 - s)*np.tanh(x)

    def step(self, dt):
        # acceleration with memory feedback and soft restoring
        restoring = - (self.omega**2) * self.f_soft(self.x)
        friction  = - self.gamma * self.v
        memory_fb = self.gain * (self.m - self.x)
        xi = self.rng.normal(0.0, self.noise)
        a = restoring + friction + memory_fb + xi
        # integrate
        self.v += a * dt
        self.x += self.v * dt
        self.m += (dt / self.tau) * (self.x - self.m)

    def run(self, steps=3000, dt=0.01, burn_in=200):
        xs, vs, ms = [], [], []
        for k in range(steps):
            self.step(dt)
            if k >= burn_in:
                xs.append(self.x); vs.append(self.v); ms.append(self.m)
        return np.array(xs), np.array(vs), np.array(ms)

# --------- Presets (interpretations) ---------
PRESETS = {
    # Lightly damped, strong memory → sustained predictive echo (e.g., cortical loop)
    "neuron": dict(memory=0.35, softness=0.65, omega=2.8, gamma=0.08, gain=0.70, noise=0.01),
    # Heavier inertia, gentle memory → slow drift to attractor (e.g., social loop)
    "social": dict(memory=0.60, softness=0.80, omega=1.2, gamma=0.12, gain=0.40, noise=0.005),
    # High Q, weak damping, moderate memory (e.g., BH accretion/growth toy)
    "cosmic": dict(memory=0.45, softness=0.55, omega=1.8, gamma=0.05, gain=0.55, noise=0.002),
    # A slightly brittle regime: low softness (more nonlinearity), can bifurcate
    "actor":  dict(memory=0.30, softness=0.30, omega=2.2, gamma=0.10, gain=0.75, noise=0.02),
}

# --------- Metrics ---------
def summarize(x, v, dt):
    import numpy as np
    N = len(x)
    rms = float(np.sqrt(np.mean(x**2)))
    mean = float(np.mean(x))
    # dominant frequency via simple FFT peak (excluding DC)
    xf = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(N, d=dt)
    if len(freqs) > 1:
        peak_idx = np.argmax(np.abs(xf)[1:]) + 1
        f_dom = float(freqs[peak_idx])
        amp_dom = float(np.abs(xf[peak_idx]) / N)
    else:
        f_dom, amp_dom = 0.0, 0.0
    # stability heuristic
    tail = x[int(0.9*N):] if N>10 else x
    var_total = float(np.var(x)) if N>1 else 0.0
    var_tail = float(np.var(tail)) if len(tail)>1 else 0.0
    stability = "bounded" if (np.max(np.abs(x)) < 4.0 * (rms + 1e-9) and var_tail < 2.0 * (var_total + 1e-12)) else "unstable"
    return dict(rms=rms, mean=mean, f_dom=f_dom, amp_dom=amp_dom, stability=stability)

# --------- Plotters ---------
def plot_time_series(t, x, prefix, outdir):
    plt.figure(figsize=(8,4.5))
    plt.plot(t, x, lw=1.2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Resonant Time Series")
    path = os.path.join(outdir, f"{prefix}_timeseries.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_phase(x, v, prefix, outdir):
    plt.figure(figsize=(5,5))
    plt.plot(x, v, lw=0.9)
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Phase Space (x vs v)")
    path = os.path.join(outdir, f"{prefix}_phase.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_spectrum(x, dt, prefix, outdir):
    N = len(x)
    xf = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(N, d=dt)
    plt.figure(figsize=(8,4.5))
    plt.plot(freqs, np.abs(xf)/N, lw=1.0)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("amplitude")
    plt.title("Spectrum |FFT|")
    path = os.path.join(outdir, f"{prefix}_spectrum.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="neuron")
    ap.add_argument("--memory", type=float, default=None)
    ap.add_argument("--softness", type=float, default=None)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--prefix", type=str, default="phase10_run")
    ap.add_argument("--outdir", type=str, default="outputs/phase10")
    args = ap.parse_args()

    # build params
    cfg = dict(**PRESETS[args.preset])
    for k in ["memory","softness","omega","gamma","gain","noise"]:
        v = getattr(args, k)
        if v is not None: cfg[k] = v

    os.makedirs(args.outdir, exist_ok=True)

    op = ResonantOperator(**cfg)
    x, v, m = op.run(steps=args.steps, dt=args.dt, burn_in=args.burn_in)
    t = np.arange(len(x)) * args.dt

    # plots
    p1 = plot_time_series(t, x, args.prefix, args.outdir)
    p2 = plot_phase(x, v, args.prefix, args.outdir)
    p3 = plot_spectrum(x, args.dt, args.prefix, args.outdir)

    # summary
    summary = dict(
        preset=args.preset,
        params=cfg,
        steps=args.steps,
        dt=args.dt,
        burn_in=args.burn_in,
        metrics=summarize(x, v, args.dt),
        outputs=dict(timeseries=p1, phase=p2, spectrum=p3),
    )
    js_path = os.path.join(args.outdir, f"{args.prefix}_summary.json")
    with open(js_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__": 
    main()
