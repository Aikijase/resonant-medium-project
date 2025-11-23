#!/usr/bin/env python3
"""
Phase 10 — Arnold Tongue Mapper (no patching)
Runs grids of (Kphi, eps) (optionally over kx, kv, omega2) by calling
tools/phase10_phasecouple_demo.py and writes:
  - outputs/phase10/<prefix>_tongue.csv
  - outputs/phase10/<prefix>_syncindex_heatmap.png
"""
import os, sys, csv, json, itertools, subprocess
import numpy as np
import matplotlib.pyplot as plt

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "phase10_phasecouple_demo.py")
OUTDIR = os.path.join(os.path.dirname(HERE), "outputs", "phase10")
os.makedirs(OUTDIR, exist_ok=True)

def run_one(args, prefix):
    cmd = [PY, RUN] + args + ["--prefix", prefix]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "runner error")
    return json.loads(p.stdout)

def main():
    # ----- CONFIGURE YOUR SLICE HERE -----
    preset = "neuron"
    omega2 = 2.4         # detuned loop-2 freq
    kv, kx = 0.10, 0.15  # coupling (vel/pos)
    steps, dt, burn = 15000, 0.01, 300
    noise = 0.01
    seed1, seed2 = 1, 2
    adapt_every = 10

    KPHI = [0.2, 0.3, 0.4, 0.5, 0.7]
    EPS  = [0.02, 0.04, 0.06, 0.08]

    # ------------------------------------
    rows = []
    for Kphi, eps in itertools.product(KPHI, EPS):
        args = [
            "--preset", preset,
            "--omega2", str(omega2),
            "--kv", str(kv), "--kx", str(kx),
            "--Kphi", str(Kphi), "--eps", str(eps),
            "--adapt_every", str(adapt_every),
            "--noise", str(noise),
            "--seed1", str(seed1), "--seed2", str(seed2),
            "--steps", str(steps), "--dt", str(dt), "--burn_in", str(burn),
            "--outdir", OUTDIR,
        ]
        prefix = f"map_w{omega2}_K{Kphi}_E{eps}".replace(".","p")
        js = run_one(args, prefix)
        m = js["metrics"]
        rows.append(dict(
            omega2=omega2, kv=kv, kx=kx, Kphi=Kphi, eps=eps,
            steps=steps, adapt_every=adapt_every, noise=noise,
            sync_index=m["sync_index"],
            mean_abs_phase_diff=m["mean_abs_phase_diff"],
            mean_abs_xdiff=m["mean_abs_xdiff"],
            timeseries=js["outputs"]["timeseries"],
            phase_diff=js["outputs"]["phase_diff"],
            x1_vs_x2=js["outputs"]["x1_vs_x2"],
        ))

    # CSV
    csv_path = os.path.join(OUTDIR, f"tongue_w{str(omega2).replace('.','p')}_kx{str(kx).replace('.','p')}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("CSV saved:", csv_path)

    # Heatmap (Kphi vs eps) of sync_index
    # build matrix consistent with the loops above (Kphi as rows, eps as cols)
    Klist = sorted(set(r["Kphi"] for r in rows))
    Elist = sorted(set(r["eps"] for r in rows))
    Z = np.zeros((len(Klist), len(Elist)))
    for r in rows:
        i = Klist.index(r["Kphi"])
        j = Elist.index(r["eps"])
        Z[i,j] = r["sync_index"]

    plt.figure(figsize=(6,5))
    plt.imshow(Z, origin="lower", aspect="auto",
               extent=[min(Elist), max(Elist), min(Klist), max(Klist)])
    plt.colorbar(label="sync_index")
    plt.xlabel("eps (adaptive pull)")
    plt.ylabel("Kphi (phase coupling)")
    plt.title(f"Arnold Tongue @ omega2={omega2}, kx={kx}, kv={kv}")
    png_path = os.path.join(OUTDIR, f"tongue_w{str(omega2).replace('.','p')}_kx{str(kx).replace('.','p')}_heatmap.png")
    plt.tight_layout(); plt.savefig(png_path, dpi=180); plt.close()
    print("Heatmap saved:", png_path)

    # Print best combo
    best = max(rows, key=lambda r: r["sync_index"])
    print("Best:", dict(Kphi=best["Kphi"], eps=best["eps"], sync_index=best["sync_index"],
                        mean_abs_phase_diff=best["mean_abs_phase_diff"]))

if __name__ == "__main__":
    main()
