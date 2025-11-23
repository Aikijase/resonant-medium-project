#!/usr/bin/env python3
"""
Phase 14 — Lockmap Sweep

Sweeps (omega2, Kphi) space using your existing Phase-10 demo
(`tools/phase10_phasecouple_demo.py`) to estimate the sync index (Si).

Outputs:
- CSV with one row per (omega2, Kphi)
- 2D PNG heatmap (matplotlib, no seaborn)
- Simple TXT run metadata

Example:
  python3 tools/phase14_lockmap.py \
    --omega2-min 2.20 --omega2-max 3.20 --omega2-step 0.02 \
    --Kphi-min 0.40 --Kphi-max 1.60 --Kphi-step 0.02 \
    --preset neuron --kv 0.10 --kx 0.20 --eps 0.06 \
    --noise 0.01 --steps 12000 --burn-in 300 \
    --outdir outputs/phase14 --prefix p14_lockmap
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, math, os, subprocess, sys
from pathlib import Path

PY = sys.executable
DEMO = Path("tools/phase10_phasecouple_demo.py")  # existing script

def run_demo(omega2: float, Kphi: float, args: argparse.Namespace):
    """Call the phase10 demo and return Si (sync_index) or None on failure."""
    cmd = [
        PY, str(DEMO),
        "--preset", args.preset,
        "--omega2", f"{omega2}",
        "--kv", f"{args.kv}",
        "--kx", f"{args.kx}",
        "--Kphi", f"{Kphi}",
        "--eps", f"{args.eps}",
        "--adapt_every", f"{args.adapt_every}",
        "--noise", f"{args.noise}",
        "--steps", f"{args.steps}",
        "--burn_in", f"{args.burn_in}",
        "--prefix", f"{args.prefix}_w{omega2:.3f}_K{Kphi:.3f}"
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=args.timeout)
        # phase10 demo prints JSON on stdout
        data = json.loads(p.stdout)
        si = float(data["metrics"]["sync_index"])
        return si, p.stdout
    except Exception as e:
        return None, f"ERROR: {e}"

def arange(start: float, stop: float, step: float):
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    for i in range(n):
        yield start + i * step

def write_png_grid(csv_path: Path, png_path: Path):
    # Build image from CSV values; keep it dependency-light
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((float(r["omega2"]), float(r["Kphi"]), float(r["Si"]) if r["Si"] else float("nan")))
    if not rows:
        return
    omega_vals = sorted(sorted(set(r[0] for r in rows)))
    K_vals = sorted(sorted(set(r[1] for r in rows)))
    grid = np.empty((len(K_vals), len(omega_vals)))
    grid[:] = np.nan

    idx_o = {v:i for i,v in enumerate(omega_vals)}
    idx_k = {v:i for i,v in enumerate(K_vals)}
    for (o,k,si) in rows:
        grid[idx_k[k], idx_o[o]] = si

    plt.figure(figsize=(8,6))
    im = plt.imshow(grid, origin="lower", aspect="auto",
                    extent=[min(omega_vals), max(omega_vals), min(K_vals), max(K_vals)])
    plt.colorbar(im, label="Sync Index (Si)")
    plt.xlabel("omega2")
    plt.ylabel("Kphi")
    plt.title("Phase 14 — Lockmap (Si)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 14 — Lockmap Sweep")
    ap.add_argument("--omega2-min", type=float, default=2.40)
    ap.add_argument("--omega2-max", type=float, default=3.00)
    ap.add_argument("--omega2-step", type=float, default=0.02)
    ap.add_argument("--Kphi-min", type=float, default=0.50)
    ap.add_argument("--Kphi-max", type=float, default=1.40)
    ap.add_argument("--Kphi-step", type=float, default=0.02)

    # Demo controls
    ap.add_argument("--preset", default="neuron")
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--adapt_every", type=int, default=15)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=90)

    # Output
    ap.add_argument("--outdir", type=Path, default=Path("outputs/phase14"))
    ap.add_argument("--prefix", default="p14_lockmap")

    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    assert DEMO.exists(), f"Missing: {DEMO} (Phase-10 demo script)."

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = args.outdir / f"{args.prefix}_{ts}.csv"
    meta_path = args.outdir / f"{args.prefix}_{ts}.meta.txt"
    png_path = args.outdir / f"{args.prefix}_{ts}.png"

    # Write meta
    meta = {
        "when": ts,
        "ranges": {
            "omega2": [args.omega2_min, args.omega2_max, args.omega2_step],
            "Kphi": [args.Kphi_min, args.Kphi_max, args.Kphi_step],
        },
        "demo": {
            "preset": args.preset, "kv": args.kv, "kx": args.kx, "eps": args.eps,
            "adapt_every": args.adapt_every, "noise": args.noise,
            "steps": args.steps, "burn_in": args.burn_in
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Sweep + CSV
    with csv_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["omega2", "Kphi", "Si", "stdout_json"])
        ovals = list(arange(args.omega2_min, args.omega2_max, args.omega2_step))
        kvals = list(arange(args.Kphi_min, args.Kphi_max, args.Kphi_step))
        total = len(ovals) * len(kvals)
        n = 0
        for K in kvals:
            for o in ovals:
                si, out = run_demo(o, K, args)
                wr.writerow([f"{o:.6f}", f"{K:.6f}", f"{si:.6f}" if si is not None else "", out.strip()[:2000]])
                n += 1
                if n % 50 == 0:
                    print(f"[p14] progress: {n}/{total}")

    # Heatmap
    try:
        write_png_grid(csv_path, png_path)
        print(f"[p14] wrote: {csv_path}")
        print(f"[p14] wrote: {png_path}")
        print(f"[p14] wrote: {meta_path}")
    except Exception as e:
        print(f"[p14] PNG generation failed: {e}")
        print(f"[p14] wrote: {csv_path}")
        print(f"[p14] wrote: {meta_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
