#!/usr/bin/env python3
"""
Phase 14 — Zoom Runner (fixed version)

Refine the lockmap around either:
  (A) a region from p14_regions.csv (via --regions-csv --region-id)
  (B) a manual center (via --center-omega2 --center-Kphi)

Outputs CSV/PNG/meta like phase14_lockmap.py, but with a tight grid.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, math, subprocess, sys
from pathlib import Path

PY = sys.executable
DEMO = Path("tools/phase10_phasecouple_demo.py")  # existing

def arange(start: float, stop: float, step: float):
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    for i in range(n):
        yield start + i * step

def run_demo(omega2: float, Kphi: float, args: argparse.Namespace):
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
        "--prefix", f"{args.prefix}_w{omega2:.4f}_K{Kphi:.4f}",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=args.timeout)
    try:
        data = json.loads(p.stdout)
        si = float(data["metrics"]["sync_index"])
    except Exception:
        si = float("nan")
    return si, p.stdout[:2000]

def read_region(regions_csv: Path, rid: int):
    with regions_csv.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if int(r["region_id"]) == rid:
                return {
                    "omin": float(r["omega2_min"]),
                    "omax": float(r["omega2_max"]),
                    "kmin": float(r["Kphi_min"]),
                    "kmax": float(r["Kphi_max"]),
                    "oc": float(r["centroid_omega2"]),
                    "kc": float(r["centroid_Kphi"]),
                }
    raise ValueError(f"region_id {rid} not found in {regions_csv}")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 14 — Zoom Runner")

    # Choose region or manual center
    ap.add_argument("--regions-csv", type=Path, default=None)
    ap.add_argument("--region-id", type=int, default=0)
    ap.add_argument("--center-omega2", type=float, default=None)
    ap.add_argument("--center-Kphi", type=float, default=None)

    # Span/pad & steps
    ap.add_argument("--omega2-span", type=float, default=None,
                    help="Full span around center for omega2 (if manual center)")
    ap.add_argument("--Kphi-span", type=float, default=None,
                    help="Full span around center for Kphi (if manual center)")
    ap.add_argument("--omega2-pad", type=float, default=0.05,
                    help="Pad added to region bbox (if region mode)")
    ap.add_argument("--Kphi-pad", type=float, default=0.05,
                    help="Pad added to region bbox (if region mode)")
    ap.add_argument("--omega2-step", type=float, default=0.005)
    ap.add_argument("--Kphi-step", type=float, default=0.005)

    # Demo controls
    ap.add_argument("--preset", default="neuron")
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--adapt_every", type=int, default=15)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--burn_in", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=120)

    # Output
    ap.add_argument("--outdir", type=Path, default=Path("outputs/phase14"))
    ap.add_argument("--prefix", default="p14_zoom")

    args = ap.parse_args(argv)
    assert DEMO.exists(), f"Missing: {DEMO}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Determine box
    if args.regions_csv and args.region_id != 0:
        box = read_region(args.regions_csv, args.region_id)
        omin = box["omin"] - args.omega2_pad
        omax = box["omax"] + args.omega2_pad
        kmin = box["kmin"] - args.Kphi_pad
        kmax = box["kmax"] + args.Kphi_pad
    else:
        if args.center_omega2 is None or args.center_Kphi is None:
            raise ValueError("Provide --regions-csv+--region-id OR --center-omega2+--center-Kphi")
        ospan = args.omega2_span if args.omega2_span is not None else 0.12
        kspan = args.Kphi_span if args.Kphi_span is not None else 0.12
        omin = args.center_omega2 - 0.5 * ospan
        omax = args.center_omega2 + 0.5 * ospan
        kmin = args.center_Kphi - 0.5 * kspan
        kmax = args.center_Kphi + 0.5 * kspan

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = args.outdir / f"{args.prefix}_{ts}.csv"
    meta_path = args.outdir / f"{args.prefix}_{ts}.meta.txt"
    png_path = args.outdir / f"{args.prefix}_{ts}.png"

    meta = {
        "mode": "region" if (args.regions_csv and args.region_id != 0) else "manual",
        "box": {"omega2": [omin, omax, args.omega2_step],
                "Kphi": [kmin, kmax, args.Kphi_step]},
        "demo": {"preset": args.preset, "kv": args.kv, "kx": args.kx, "eps": args.eps,
                 "adapt_every": args.adapt_every, "noise": args.noise,
                 "steps": args.steps, "burn_in": args.burn_in}
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Sweep
    with csv_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["omega2", "Kphi", "Si", "stdout_json"])
        ovals = list(arange(omin, omax, args.omega2_step))
        kvals = list(arange(kmin, kmax, args.Kphi_step))
        total = len(ovals) * len(kvals)
        n = 0
        for K in kvals:
            for o in ovals:
                si, out = run_demo(o, K, args)
                wr.writerow([f"{o:.6f}", f"{K:.6f}",
                             f"{si:.6f}" if si==si else "",
                             out.strip()])
                n += 1
                if n % 100 == 0:
                    print(f"[zoom] progress: {n}/{total}")

    # Quick heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        rows = []
        with csv_path.open() as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["Si"]:
                    rows.append((float(r["omega2"]), float(r["Kphi"]), float(r["Si"])))
        if rows:
            O = sorted({r[0] for r in rows})
            K = sorted({r[1] for r in rows})
            idx_o = {v:i for i,v in enumerate(O)}
            idx_k = {v:i for i,v in enumerate(K)}
            grid = np.full((len(K), len(O)), np.nan)
            for o,k,si in rows:
                grid[idx_k[k], idx_o[o]] = si
            plt.figure(figsize=(8,6))
            im = plt.imshow(grid, origin="lower", aspect="auto",
                            extent=[min(O), max(O), min(K), max(K)])
            plt.colorbar(im, label="Sync Index (Si)")
            plt.xlabel("omega2")
            plt.ylabel("Kphi")
            plt.title("Phase 14 — Zoom")
            plt.tight_layout()
            plt.savefig(png_path, dpi=160)
            plt.close()
        print(f"[zoom] wrote: {csv_path}")
        print(f"[zoom] wrote: {png_path}")
        print(f"[zoom] wrote: {meta_path}")
    except Exception as e:
        print(f"[zoom] PNG failed: {e}")
        print(f"[zoom] wrote: {csv_path}")
        print(f"[zoom] wrote: {meta_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
