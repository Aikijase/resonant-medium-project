#!/usr/bin/env python3
"""
Phase-19 (Sim): g×κ toy cross-correlation under resonant-memory modulation.

- Drop-in, read-only helper (no edits to existing code)
- Headless-safe (Agg backend), re-runnable, writes explicit artifacts
- Defaults to Option A: omega0 ≈ 0.5, Q ≈ 0.4 (low-frequency memory)

Outputs (always in outputs/phase19/):
  - gxk_sim_summary.csv     (CSV with key scalars across sweeps)
  - gxk_sim_report.txt      (human-readable text)
  - gxk_sim_plot.png        (ΔC_ell / C_ell vs parameter; plus spectra)

Usage examples:
  python3 tools/phase19_gxk_sim.py
  python3 tools/phase19_gxk_sim.py --omega0 0.5 --Q 0.4 --alpha 0.12
  python3 tools/phase19_gxk_sim.py --sweep --omega0 0.35 0.55 --Q 0.3 0.5 --alpha 0.04 0.20

Interpretation:
  We build a toy matter P(k) and lens/gal kernels, then imprint a Lorentzian
  “memory notch/boost” around omega0 with width set by Q. We integrate a
  Limber-lite proxy for g×κ amplitude and compare to baseline to get
  ΔC_ell/C_ell. This is qualitative, fast, and good enough to verify the
  predicted suppression → reversal trend without heavy sky data.
"""
import argparse, os, sys, math, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------
def lorentz_profile(k, k0, Q):
    """Unit-peak Lorentzian centered at k0 with HWHM = k0/(2Q)."""
    gamma = k0/(2.0*Q) if Q>0 else 1e-6
    return 1.0 / (1.0 + ((k - k0)/gamma)**2)

def toy_pk(k, A=1.0, ns=0.96, kc=1.2):
    """Very light toy matter power spectrum (no BAO):
       P(k) = A * k^ns * exp(-(k/kc)^2)  (monotone roll-off)."""
    return A * np.power(np.clip(k,1e-6,None), ns) * np.exp(-(k/kc)**2)

def n_of_z(z, z0=0.5):
    """Toy source/galaxy redshift distribution ~ z^2 exp(-(z/z0)^1.5)."""
    return z**2 * np.exp(- (z/z0)**1.5)

def W_kappa(z, z_s=1.0):
    """Toy lensing efficiency kernel ~ z (1 - z/z_s) for z<z_s, else 0."""
    w = z*(1.0 - z/z_s)
    w[z>z_s] = 0.0
    return np.clip(w, 0, None)

def W_gal(z, bias=1.2, z0=0.5):
    """Toy galaxy kernel ~ bias * n(z)."""
    return bias * n_of_z(z, z0=z0)

def gxk_amplitude(k, pk, z, Wg, Wk):
    """Simple separable proxy for C_ell amplitude:
       A ∝ ∫dz Wg(z)Wk(z) * ∫dk pk(k) (normalized)."""
    Iz = np.trapz(Wg*Wk, z)
    Ik = np.trapz(pk, k)
    return Iz * Ik

def imprint_memory(pk, k, k0, Q, alpha, kind="notch"):
    """
    Apply a Lorentzian modulation around k0 with strength alpha:
      notch: P -> P * (1 - alpha * L)
      boost: P -> P * (1 + alpha * L)
    """
    L = lorentz_profile(k, k0, Q)
    mod = 1.0 + (alpha*L if kind=="boost" else -alpha*L)
    return np.clip(pk * mod, 0.0, None)

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega0", type=float, default=0.5, help="Resonant center k0 (~ω0)")
    ap.add_argument("--Q",      type=float, default=0.4, help="Quality factor")
    ap.add_argument("--alpha",  type=float, default=0.10, help="Memory strength (0..~0.3)")
    ap.add_argument("--kind",   type=str,   default="notch", choices=["notch","boost"], help="Modulation type")
    ap.add_argument("--sweep",  action="store_true", help="Sweep ranges instead of single eval")
    ap.add_argument("--omega0-range", nargs=2, type=float, metavar=("MIN","MAX"))
    ap.add_argument("--Q-range",      nargs=2, type=float, metavar=("MIN","MAX"))
    ap.add_argument("--alpha-range",  nargs=2, type=float, metavar=("MIN","MAX"))
    ap.add_argument("--n-steps", type=int, default=7, help="Steps per parameter when sweeping")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = "outputs/phase19"
    safe_mkdir(outdir)

    # k, z grids (lightweight)
    k = np.linspace(0.01, 3.0, 2200)        # h/Mpc-ish units (toy)
    z = np.linspace(0.00, 2.0, 800)

    # Kernels & baseline
    pk0 = toy_pk(k)
    Wg  = W_gal(z, bias=1.2, z0=0.5)
    Wk  = W_kappa(z, z_s=1.0)
    A0  = gxk_amplitude(k, pk0, z, Wg, Wk)  # baseline amplitude
    if not np.isfinite(A0) or A0<=0:
        print("ERROR: baseline amplitude invalid.", file=sys.stderr)
        sys.exit(1)

    rows = []
    def eval_point(k0, Q, alpha, kind):
        pk_mod = imprint_memory(pk0, k, k0, Q, alpha, kind=kind)
        A1 = gxk_amplitude(k, pk_mod, z, Wg, Wk)
        delta = (A1 - A0) / A0
        return float(A1), float(delta)

    if args.sweep:
        k0_min, k0_max = (args.omega0_range or [args.omega0*0.7, args.omega0*1.3])
        Q_min,  Q_max  = (args.Q_range      or [max(0.15, args.Q*0.75), args.Q*1.25])
        a_min,  a_max  = (args.alpha_range  or [max(0.02, args.alpha*0.5), min(0.3, args.alpha*1.5)])

        k0_grid = np.linspace(k0_min, k0_max, args.n_steps)
        Q_grid  = np.linspace(Q_min, Q_max, args.n_steps)
        a_grid  = np.linspace(a_min, a_max, args.n_steps)

        for k0 in k0_grid:
            for Q in Q_grid:
                for a in a_grid:
                    A1, d = eval_point(k0, Q, a, args.kind)
                    rows.append(dict(omega0=k0, Q=Q, alpha=a, kind=args.kind, A=A1, dA_over_A=d))
    else:
        A1, d = eval_point(args.omega0, args.Q, args.alpha, args.kind)
        rows.append(dict(omega0=args.omega0, Q=args.Q, alpha=args.alpha, kind=args.kind, A=A1, dA_over_A=d))

    df = pd.DataFrame(rows).sort_values(["omega0","Q","alpha"]).reset_index(drop=True)

    # Lightweight PASS/FAIL: we expect a clear negative ΔC/C (~-few % to -tens %) for notch at Option A.
    # We'll call PASS if median ΔC/C is within [-0.25, -0.01].
    med = float(np.median(df["dA_over_A"]))
    expected_ok = (-0.25 <= med <= -0.01) if (df["kind"].iloc[0] == "notch") else (0.01 <= med <= 0.25)
    status = "PASS" if expected_ok else "WARN"

    # Plots
    plt.figure(figsize=(7.5, 5.0), dpi=140)
    if args.sweep:
        # Heatmap-like scatter of ΔC/C across parameters
        sc = plt.scatter(df["omega0"], df["Q"], c=df["dA_over_A"], s=32, edgecolor="none")
        cbar = plt.colorbar(sc, label="ΔC_ℓ / C_ℓ (proxy)")
        plt.xlabel("ω₀ (k0)")
        plt.ylabel("Q")
        plt.title("Phase-19 (Sim): ΔC_ℓ/C_ℓ across (ω₀, Q)\n(kind: %s, α swept)" % df["kind"].iloc[0])
    else:
        plt.plot(df["alpha"], df["dA_over_A"], marker="o")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("α (memory strength)")
        plt.ylabel("ΔC_ℓ / C_ℓ (proxy)")
        plt.title("Phase-19 (Sim): ΔC_ℓ/C_ℓ vs α\n(ω₀=%.3f, Q=%.3f, kind=%s)" % (df["omega0"].iloc[0], df["Q"].iloc[0], df["kind"].iloc[0]))
    plt.tight_layout()
    plot_path = os.path.join(outdir, "gxk_sim_plot.png")
    plt.savefig(plot_path)

    # Summary CSV
    csv_path = os.path.join(outdir, "gxk_sim_summary.csv")
    df.to_csv(csv_path, index=False)

    # Text report
    rep = []
    rep.append("Phase-19 (Sim) g×κ under resonant memory")
    rep.append("Outdir: %s" % outdir)
    rep.append("Mode: %s" % ("SWEEP" if args.sweep else "SINGLE"))
    rep.append("Median ΔC_ℓ/C_ℓ: %.4f" % med)
    rep.append("Expected trend: %s" % ("suppression (negative)" if df['kind'].iloc[0]=="notch" else "enhancement (positive)"))
    rep.append("Status: %s" % status)
    rep.append("")
    rep.append("Top 5 | strongest magnitude | rows:")
    top = df.reindex(df["dA_over_A"].abs().sort_values(ascending=False).index).head(5)
    rep.append(top.to_string(index=False))
    rep_path = os.path.join(outdir, "gxk_sim_report.txt")
    with open(rep_path, "w") as f:
        f.write("\n".join(rep))

    print(f"Wrote {csv_path}")
    print(f"Wrote {rep_path}")
    print(f"Wrote {plot_path}")
    print(f"RESULT: {status}")

if __name__ == "__main__":
    main()
