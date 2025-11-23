#!/usr/bin/env python3
"""
Phase 14 — Region Finder (contiguous lock regions)

Reads one or more Phase-14 lockmap CSVs (columns: omega2, Kphi, Si),
grids them, thresholds at --min-si, finds 4-neighbor contiguous regions,
and writes:
  - <outdir>/<prefix>_regions.csv     (one row per region: stats, bbox, centroid)
  - <outdir>/<prefix>_regions.png     (visualization with region IDs)
  - <outdir>/<prefix>_regions.meta.txt (parameters used)

Dependencies: numpy, matplotlib (no seaborn)

Example:
  python3 tools/phase14_regions.py \
    --glob "outputs/phase14/p14_*.csv" \
    --min-si 0.95 \
    --outdir outputs/phase14 \
    --prefix p14_regions
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

def read_points(files: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique sorted omega2 grid, Kphi grid, and Si grid (with NaNs)."""
    pts = []
    for fp in files:
        with fp.open() as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    o = float(r["omega2"])
                    k = float(r["Kphi"])
                    si = r.get("Si","").strip()
                    si = float(si) if si != "" else float("nan")
                    pts.append((o,k,si))
                except Exception:
                    continue
    if not pts:
        raise RuntimeError("No points found in CSVs")

    omegas = sorted({p[0] for p in pts})
    ks     = sorted({p[1] for p in pts})
    O = np.array(omegas)
    K = np.array(ks)

    # map (o,k)->si
    index = {(o,k): si for (o,k,si) in pts}
    grid = np.full((len(ks), len(omegas)), np.nan, dtype=float)  # rows=K, cols=O
    for i,k in enumerate(ks):
        for j,o in enumerate(omegas):
            if (o,k) in index:
                grid[i,j] = index[(o,k)]
    return O, K, grid

def label_regions(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-neighbor component labeling. Returns (labels, nlabels)."""
    h, w = mask.shape
    labels = np.zeros((h,w), dtype=np.int32)
    current = 0
    # simple BFS
    from collections import deque
    for i in range(h):
        for j in range(w):
            if not mask[i,j] or labels[i,j] != 0:
                continue
            current += 1
            q = deque([(i,j)])
            labels[i,j] = current
            while q:
                y,x = q.popleft()
                for (yy,xx) in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= yy < h and 0 <= xx < w:
                        if mask[yy,xx] and labels[yy,xx] == 0:
                            labels[yy,xx] = current
                            q.append((yy,xx))
    return labels, current

def region_stats(O: np.ndarray, K: np.ndarray, grid: np.ndarray,
                 labels: np.ndarray, rid: int) -> Dict:
    """Compute stats for a given region id."""
    ys, xs = np.where(labels == rid)
    if len(ys) == 0:
        return {}
    sis = grid[ys, xs]
    # bbox in index space
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # centroid (index-weighted)
    cy = ys.mean()
    cx = xs.mean()
    # map to coordinate space
    bbox = {
        "Kphi_min": float(K[y0]),
        "Kphi_max": float(K[y1]),
        "omega2_min": float(O[x0]),
        "omega2_max": float(O[x1]),
        "width_cells": int(x1 - x0 + 1),
        "height_cells": int(y1 - y0 + 1),
    }
    centroid = {
        "omega2": float(O[int(round(cx))]),
        "Kphi":   float(K[int(round(cy))]),
    }
    return {
        "region_id": rid,
        "n_cells": int(len(ys)),
        "si_min": float(np.nanmin(sis)),
        "si_max": float(np.nanmax(sis)),
        "si_mean": float(np.nanmean(sis)),
        "bbox": bbox,
        "centroid": centroid,
    }

def draw_regions_png(O: np.ndarray, K: np.ndarray, grid: np.ndarray,
                     labels: np.ndarray, out_png: Path, min_si: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # base image
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[O.min(), O.max(), K.min(), K.max()])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sync Index (Si)")
    ax.set_xlabel("omega2")
    ax.set_ylabel("Kphi")
    ax.set_title(f"Regions (Si ≥ {min_si:g})")

    # overlay region labels (centroids)
    max_id = labels.max()
    for rid in range(1, max_id+1):
        ys, xs = np.where(labels == rid)
        if len(ys) == 0:
            continue
        cy = K[int(round(ys.mean()))]
        cx = O[int(round(xs.mean()))]
        ax.text(cx, cy, str(rid), ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main(argv=None):
    import glob
    import argparse, json
    ap = argparse.ArgumentParser(description="Phase 14 — Region Finder")
    ap.add_argument("--glob", default="outputs/phase14/p14_*.csv", help="CSV glob to read")
    ap.add_argument("--min-si", type=float, default=0.95, help="Threshold for inclusion")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/phase14"))
    ap.add_argument("--prefix", default="p14_regions")
    args = ap.parse_args(argv)

    files = sorted(Path().glob(args.glob))
    if not files:
        print(f"[regions] No files matched: {args.glob}")
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv  = args.outdir / f"{args.prefix}.csv"
    out_png  = args.outdir / f"{args.prefix}.png"
    out_meta = args.outdir / f"{args.prefix}.meta.txt"

    O, K, grid = read_points(files)
    mask = np.isfinite(grid) & (grid >= args.min_si)
    labels, nlabels = label_regions(mask)

    # Stats per region
    rows = []
    for rid in range(1, nlabels+1):
        st = region_stats(O, K, grid, labels, rid)
        if st:
            rows.append(st)

    # Write CSV
    import csv, json
    with out_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "region_id", "n_cells",
            "si_min", "si_mean", "si_max",
            "omega2_min","omega2_max","Kphi_min","Kphi_max",
            "centroid_omega2","centroid_Kphi"
        ])
        for st in rows:
            wr.writerow([
                st["region_id"], st["n_cells"],
                f"{st['si_min']:.6f}", f"{st['si_mean']:.6f}", f"{st['si_max']:.6f}",
                f"{st['bbox']['omega2_min']:.6f}", f"{st['bbox']['omega2_max']:.6f}",
                f"{st['bbox']['Kphi_min']:.6f}",  f"{st['bbox']['Kphi_max']:.6f}",
                f"{st['centroid']['omega2']:.6f}", f"{st['centroid']['Kphi']:.6f}",
            ])

    # Meta + PNG
    out_meta.write_text(json.dumps({
        "glob": args.glob,
        "min_si": args.min_si,
        "files": [str(p) for p in files],
        "grid": {"omega2_len": len(O), "Kphi_len": len(K)},
        "n_regions": len(rows)
    }, indent=2))

    try:
        draw_regions_png(O, K, grid, labels, out_png, args.min_si)
        print(f"[regions] wrote: {out_csv}")
        print(f"[regions] wrote: {out_png}")
        print(f"[regions] wrote: {out_meta}")
    except Exception as e:
        print(f"[regions] PNG failed: {e}")
        print(f"[regions] wrote: {out_csv}")
        print(f"[regions] wrote: {out_meta}")
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
