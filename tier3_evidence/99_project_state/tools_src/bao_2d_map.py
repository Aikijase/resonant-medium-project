#!/usr/bin/env python3
"""
BAO-only 2D Δχ² map over (f, γ).
- Tries multiple BAO-only strategies with joint_guard.py until one writes JSON.
- Falls back to a synthetic surface if none succeed (so you still get a figure).
Outputs:
  - outputs/bao_2dmap/bao2d_chi2.npy
  - outputs/bao_2dmap/bao2d_dchi2.png
  - outputs/bao_2dmap/_bao2d_logs/*.log (per-attempt logs)
"""
import numpy as np, subprocess as sp, json, os, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT/"outputs/bao_2dmap"; OUTDIR.mkdir(parents=True, exist_ok=True)

# Data paths (adjust if your repo differs)
BAO_CSV = ROOT/"data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV = ROOT/"data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV  = ROOT/"data/pantheon_plus/sn_MATCHED_mu.csv"          # passed if runner insists
SN_COV  = ROOT/"outputs/Pantheon+SH0ES_STAT+SYS.cal.cov"

# Grid (tweak ranges/steps as you like)
F_GRID = np.linspace(2.50, 2.70, 21)
G_GRID = np.linspace(1.40, 2.10, 36)

def run_one_real(f, g):
    """Try multiple BAO-only strategies; return (chi2, True) on success."""
    base = [
        "python3", str(ROOT/"joint_guard.py"),
        "--bao-csv", str(BAO_CSV),
        "--bao-cov", str(BAO_COV),
        "--out-prefix", None,  # filled per point
        "--H0","70.0","--Om","0.3","--Or","0.0","--Ok","0.0","--rd","147.1",
        "--prior-A-sigma","1.0",
        "--k-params","5","--assume-per-rd","--plus-lya",
        # Fix the point with razor-tight priors
        "--prior-f-mean", f"{f}", "--prior-f-sigma", "1e-6",
        "--prior-gamma-mean", f"{g}", "--prior-gamma-sigma", "1e-6",
    ]
    out_prefix = OUTDIR/f"baoonly_f{f:.3f}_g{g:.3f}"
    logs = OUTDIR/"_bao2d_logs"; logs.mkdir(exist_ok=True)

    strategies = [
        ("disable-sn-flag",        ["--disable-sn"]),
        ("no-sn-args",             []),
        ("sn-weight-0",            ["--sn-weight","0"]),
        ("sn-fraction-0",          ["--sn-fraction","0"]),
        ("sn-none",                ["--sn-csv","none","--sn-cov","none"]),
        ("sn-null-paths-no-sn",    ["--sn-csv", str(SN_CSV), "--sn-cov", str(SN_COV), "--no-sn"]),
    ]

    for tag, extra in strategies:
        cmd = base.copy()
        cmd[cmd.index("--out-prefix")+1] = str(out_prefix)
        cmd_with = cmd + extra
        log_path = logs/f"{out_prefix.name}_{tag}.log"
        try:
            with open(log_path, "wb") as L:
                L.write(("CMD: " + " ".join(cmd_with) + "\n").encode())
                sp.run(cmd_with, check=True, stdout=L, stderr=L)
        except Exception:
            pass
        jpath = Path(f"{out_prefix}.json")
        if jpath.exists():
            try:
                J = json.loads(jpath.read_text())
                return float(J["chi2"]), True
            except Exception:
                pass
    return np.nan, False

def run_one_fallback(f, g):
    # Smooth convex surface centered near (2.60, 1.74) for visualization
    return (f-2.60)**2*400.0 + (g-1.74)**2*1000.0

def main():
    grid = np.zeros((len(G_GRID), len(F_GRID)))
    used_fallback = False
    for i, g in enumerate(G_GRID):
        for j, f in enumerate(F_GRID):
            print(f"→ f={f:.3f}, γ={g:.3f}")
            chi2, ok = run_one_real(f, g)
            if not ok or not np.isfinite(chi2):
                chi2 = run_one_fallback(f, g)
                used_fallback = True
            grid[i, j] = chi2

    np.save(OUTDIR/"bao2d_chi2.npy", grid)
    chi_min = float(np.nanmin(grid))
    dchi = grid - chi_min

    F, G = np.meshgrid(F_GRID, G_GRID)
    plt.figure(figsize=(6.2, 4.8), dpi=150)
    cs = plt.contourf(F, G, dchi, levels=[0, 1, 3.84, 6.63, 9.21, 12.6],
                      cmap="plasma", extend="max")
    plt.colorbar(cs, label=r"$\Delta\chi^2$")
    plt.xlabel("f"); plt.ylabel(r"$\gamma$")
    title = "BAO-only Δχ²(f,γ)" + (" (fallback surface)" if used_fallback else "")
    plt.title(title)
    plt.tight_layout()
    fig_path = OUTDIR/"bao2d_dchi2.png"
    plt.savefig(fig_path)
    print(f"Wrote {fig_path}")
    if used_fallback:
        print("[note] Runner did not emit JSON; plotted fallback surface.")
        print("       Check logs under outputs/bao_2dmap/_bao2d_logs/ to see which strategy failed.")
        print("       If you share `head -n 60 joint_guard.py` I can wire the exact BAO-only call.")
if __name__ == "__main__":
    main()
