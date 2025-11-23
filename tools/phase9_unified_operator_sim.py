#!/usr/bin/env python3
"""
Phase 9 — Unified Resonant Operator (URO) Simulator
---------------------------------------------------
Single drop-in script to simulate the operator:

    α ẍ + β ẋ + γ (x - x*(t)) = F sin(ω t) + ξ(t)

across three domains by swapping coefficient sets and targets via YAML config.

Outputs:
  - <out_prefix>.csv   : t, x, v, x_star, err, beta_eff, energy
  - <out_prefix>_timeseries.png
  - <out_prefix>_phase.png
  - <out_prefix>_config.echo.yaml   (the parsed/expanded config)

Dependencies: numpy, scipy, matplotlib, pyyaml
"""
import argparse, os, sys, json, math, datetime
from dataclasses import dataclass
import numpy as np

# Strictly standard libs above this line
try:
    import yaml
except ImportError as e:
    print("Missing dependency 'pyyaml'. Try: python3 -m pip install pyyaml", file=sys.stderr); raise
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Missing dependency 'matplotlib'. Try: python3 -m pip install matplotlib", file=sys.stderr); raise
try:
    from scipy.integrate import solve_ivp
except ImportError as e:
    print("Missing dependency 'scipy'. Try: python3 -m pip install scipy", file=sys.stderr); raise


# -----------------------------
# Utilities
# -----------------------------
def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# -----------------------------
# Target x*(t) definitions
# -----------------------------
def build_xstar_fn(cfg_target: dict):
    """
    Supported forms:
      type: constant
        value: <float>

      type: sinusoid
        A: <float>           (amplitude)
        omega: <float>       (angular frequency)
        phi: <float>         (phase, radians) [optional]
        offset: <float>      (DC offset) [optional]

      type: step
        t0: <float>          (switch time)
        value0: <float>
        value1: <float>
    """
    ttype = (cfg_target or {}).get("type", "constant").lower()

    if ttype == "constant":
        val = float(cfg_target.get("value", 0.0))
        return lambda t: val

    if ttype == "sinusoid":
        A = float(cfg_target.get("A", 0.0))
        om = float(cfg_target.get("omega", 1.0))
        phi = float(cfg_target.get("phi", 0.0))
        off = float(cfg_target.get("offset", 0.0))
        return lambda t: off + A * np.sin(om * t + phi)

    if ttype == "step":
        t0 = float(cfg_target.get("t0", 0.0))
        v0 = float(cfg_target.get("value0", 0.0))
        v1 = float(cfg_target.get("value1", 1.0))
        return lambda t: v0 if t < t0 else v1

    raise ValueError(f"Unsupported target type: {ttype}")


# -----------------------------
# Adaptation (optional)
# -----------------------------
@dataclass
class BetaAdapt:
    base_beta: float
    k: float = 0.0            # strength of adaptation (>=0)
    target_err: float = 0.1   # error scale for adaptation
    limit_lo: float = 1e-6    # lower guard
    limit_hi: float = 1e6     # upper guard

    @staticmethod
    def from_cfg(beta: float, cfg: dict | None):
        if not cfg:
            return BetaAdapt(beta)
        return BetaAdapt(
            base_beta=float(beta),
            k=float(cfg.get("k", 0.0)),
            target_err=float(cfg.get("target_error", 0.1)),
            limit_lo=float(cfg.get("limit_lo", 1e-6)),
            limit_hi=float(cfg.get("limit_hi", 1e6)),
        )

    def beta_eff(self, err: float) -> float:
        """
        Simple tempo-like adaptation:
          β_eff = β * (1 + k * tanh(|err|/target_err - 1))
        """
        if self.k <= 0.0:
            return self.base_beta
        s = abs(err) / max(self.target_err, 1e-12) - 1.0
        be = self.base_beta * (1.0 + self.k * np.tanh(s))
        return float(np.clip(be, self.limit_lo, self.limit_hi))


# -----------------------------
# Noise (optional)
# -----------------------------
def build_noise_fn(cfg_noise: dict | None):
    """
    noise:
      type: "gaussian"
      std: 0.0
      seed: 123
    Added to RHS as ξ(t).
    """
    if not cfg_noise:
        return lambda t: 0.0
    ntype = str(cfg_noise.get("type", "gaussian")).lower()
    if ntype != "gaussian":
        raise ValueError(f"Unsupported noise type: {ntype}")
    std = float(cfg_noise.get("std", 0.0))
    seed = cfg_noise.get("seed", None)
    rng = np.random.default_rng(seed)
    # White-ish noise: new value each call; for small dt this is OK for toy sims.
    return lambda t: rng.normal(0.0, std) if std > 0.0 else 0.0


# -----------------------------
# RHS builder
# -----------------------------
def build_rhs(alpha, beta_adapt: BetaAdapt, gamma, xstar_fn, drive_F, drive_omega, noise_fn):
    """
    Convert 2nd order to first order system:
      y = [x, v],  ẋ = v
      v̇ = (1/α) * [ -β_eff v - γ (x - x*) + F sin(ω t) + ξ(t) ]
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    def rhs(t, y):
        x, v = y
        xstar = xstar_fn(t)
        err = (x - xstar)
        beta_eff = beta_adapt.beta_eff(err)
        forcing = drive_F * np.sin(drive_omega * t) if drive_F != 0.0 else 0.0
        acc = ( - beta_eff * v - gamma * err + forcing + noise_fn(t) ) / alpha
        return [v, acc]
    return rhs


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 9 Unified Resonant Operator simulator")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--out-prefix", default=os.path.join("outputs", "phase9", f"run_{timestamp()}"),
                    help="Output prefix (no extension). Default outputs/phase9/run_<ts>")
    ap.add_argument("--tmax", type=float, default=None, help="Override t_max (seconds)")
    ap.add_argument("--steps", type=int, default=None, help="Override number of output samples")
    args = ap.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    sim = cfg.get("sim", {})
    dom = cfg.get("domain", "unspecified")
    params = cfg.get("params", {})
    drive = cfg.get("drive", {})
    target = cfg.get("target", {})
    noise = cfg.get("noise", {})
    adaptation = cfg.get("adaptation", {})

    # Resolve scalars with defaults
    alpha = float(params.get("alpha", 1.0))
    beta  = float(params.get("beta", 0.2))
    gamma = float(params.get("gamma", 1.0))

    drive_F = float(drive.get("F", 0.0))
    drive_omega = float(drive.get("omega", 1.0))

    t0 = float(sim.get("t0", 0.0))
    t_max = float(sim.get("t_max", 40.0 if args.tmax is None else args.tmax))
    n_steps = int(sim.get("n_steps", 4000 if args.steps is None else args.steps))
    x0 = float(sim.get("x0", 0.0))
    v0 = float(sim.get("v0", 0.0))

    # Build functions
    xstar_fn = build_xstar_fn(target)
    noise_fn = build_noise_fn(noise)
    beta_adapt = BetaAdapt.from_cfg(beta, adaptation)
    rhs = build_rhs(alpha, beta_adapt, gamma, xstar_fn, drive_F, drive_omega, noise_fn)

    # Integrate
    t_eval = np.linspace(t0, t_max, n_steps)
    sol = solve_ivp(rhs, (t0, t_max), [x0, v0], t_eval=t_eval, rtol=1e-8, atol=1e-10, method="RK45")
    if not sol.success:
        print(f"[phase9] Integration failed: {sol.message}", file=sys.stderr)
        sys.exit(2)

    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    xstar = np.array([xstar_fn(tt) for tt in t], dtype=float)
    err = x - xstar
    beta_eff = np.array([beta_adapt.beta_eff(e) for e in err], dtype=float)

    # Simple energy-like metric (for intuition): E = 0.5 α v^2 + 0.5 γ (x - x*)^2
    energy = 0.5 * alpha * v**2 + 0.5 * gamma * err**2

    # Prepare outputs
    out_prefix = args.out_prefix
    ensure_parent(out_prefix + ".csv")

    # Save CSV
    import csv
    with open(out_prefix + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "v", "x_star", "err", "beta_eff", "energy"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.12g}", f"{x[i]:.12g}", f"{v[i]:.12g}",
                        f"{xstar[i]:.12g}", f"{err[i]:.12g}", f"{beta_eff[i]:.12g}", f"{energy[i]:.12g}"])
    print(f"[phase9] CSV saved: {out_prefix}.csv")

    # Save config echo
    echo = dict(cfg)  # shallow copy
    echo.setdefault("_meta", {})
    echo["_meta"]["domain"] = dom
    echo["_meta"]["resolved"] = {
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "drive_F": drive_F, "drive_omega": drive_omega,
        "t0": t0, "t_max": t_max, "n_steps": n_steps,
        "x0": x0, "v0": v0,
    }
    with open(out_prefix + "_config.echo.yaml", "w") as f:
        yaml.safe_dump(echo, f, sort_keys=False)
    print(f"[phase9] Echo config saved: {out_prefix}_config.echo.yaml")

    # Plots
    # 1) Time series
    plt.figure(figsize=(9, 4.8))
    plt.plot(t, x, label="x(t)")
    plt.plot(t, xstar, linestyle="--", label="x*(t)")
    plt.xlabel("t")
    plt.ylabel("State")
    plt.title(f"URO Time Series — {dom}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_prefix + "_timeseries.png", dpi=150)
    plt.close()
    print(f"[phase9] Plot saved: {out_prefix}_timeseries.png")

    # 2) Phase portrait
    plt.figure(figsize=(5.6, 5.6))
    plt.plot(x, v)
    plt.xlabel("x")
    plt.ylabel("v = ẋ")
    plt.title(f"URO Phase Portrait — {dom}")
    plt.tight_layout()
    plt.savefig(out_prefix + "_phase.png", dpi=150)
    plt.close()
    print(f"[phase9] Plot saved: {out_prefix}_phase.png")

    # Simple post-hoc metrics
    half = len(t) // 2
    p2p = np.max(x[half:]) - np.min(x[half:]) if half < len(x) else np.nan
    rms_err = float(np.sqrt(np.mean(err[half:]**2))) if half < len(err) else np.nan
    print(f"[phase9] Post-hoc metrics (after mid-time): peak_to_peak={p2p:.6g}   rms_err={rms_err:.6g}")

if __name__ == "__main__":
    main()
