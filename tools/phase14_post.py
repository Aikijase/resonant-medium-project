#!/usr/bin/env python3
# tools/phase14_post.py
import argparse, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Phase 14 post-processor: heatmap, masks, boundary (robust)")
    p.add_argument("--csv", required=True, help="results CSV (e.g., outputs/phase14/p14_zoom_r1_results.csv)")
    p.add_argument("--outdir", help="output directory (default: same as csv)")
    p.add_argument("--prefix", help="output prefix (default: csv basename without _results)")
    p.add_argument("--threshold", type=float, default=0.95, help="desired contour level")
    p.add_argument("--autoq", type=float, default=0.90, help="fallback quantile if threshold not in (min,max)")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()

def load_and_pivot(csv_path):
    df = pd.read_csv(csv_path)
    df["omega2"] = df["omega2"].astype(float)
    df["Kphi"] = df["Kphi"].astype(float)

    ok = df["status"].astype(str) == "ok"
    si = df.loc[ok, ["omega2","Kphi","sync_index"]].copy()
    si["sync_index"] = pd.to_numeric(si["sync_index"], errors="coerce")

    W = np.unique(df["omega2"].values); W.sort()
    K = np.unique(df["Kphi"].values); K.sort()

    grid = np.full((len(W), len(K)), np.nan, dtype=float)
    status_grid = np.full((len(W), len(K)), "", dtype=object)

    w_to_i = {w:i for i,w in enumerate(W)}
    k_to_j = {k:j for j,k in enumerate(K)}

    for _, row in df[["omega2","Kphi","status"]].iterrows():
        status_grid[w_to_i[row["omega2"]], k_to_j[row["Kphi"]]] = str(row["status"])

    for _, row in si.iterrows():
        grid[w_to_i[row["omega2"]], k_to_j[row["Kphi"]]] = row["sync_index"]

    return df, W, K, grid, status_grid

def save_grid_csv(out_grid_csv, W, K, grid):
    rows = []
    for i, w in enumerate(W):
        for j, k in enumerate(K):
            v = grid[i,j]
            rows.append({"omega2": w, "Kphi": k, "sync_index": (None if np.isnan(v) else float(v))})
    pd.DataFrame(rows).to_csv(out_grid_csv, index=False)

def plot_heatmap_with_boundary(path, title, W, K, grid, level, dpi=160):
    plt.figure(figsize=(8,6), dpi=dpi)
    im = plt.imshow(grid, origin="lower",
                    extent=[K.min(), K.max(), W.min(), W.max()],
                    aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, label="sync_index")
    plt.xlabel("Kphi")
    plt.ylabel("omega2")
    plt.title(title)

    drew_contour = False
    try:
        CS = plt.contour(K, W, grid, levels=[level], colors="white", linewidths=1.5)
        for c in CS.collections:
            c.set_label(f"level={level:.4f}")
        drew_contour = True
    except Exception:
        pass

    # If contour fails (e.g., level outside data), draw a filled mask >= level as a fallback outline
    if not drew_contour and np.isfinite(level):
        try:
            plt.contourf(K, W, np.where(grid >= level, 1.0, np.nan), levels=[0.999,1.001], alpha=0.15)
            plt.contour(K, W, np.where(grid >= level, 1.0, np.nan), levels=[1.0], colors="white", linewidths=1.0)
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_timeout_mask_png(path, title, W, K, status_grid, dpi=160):
    mask = np.zeros_like(status_grid, dtype=float)
    for i in range(status_grid.shape[0]):
        for j in range(status_grid.shape[1]):
            s = status_grid[i,j]
            mask[i,j] = 1.0 if s in ("timeout","error") else 0.0
    plt.figure(figsize=(8,6), dpi=dpi)
    im = plt.imshow(mask, origin="lower",
                    extent=[K.min(), K.max(), W.min(), W.max()],
                    aspect="auto")
    plt.colorbar(im, label="timeout/error mask (1=yes)")
    plt.xlabel("Kphi"); plt.ylabel("omega2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def extract_boundary_csv(path, W, K, grid, level):
    # Try line contour first
    try:
        fig, ax = plt.subplots()
        CS = ax.contour(K, W, grid, levels=[level])
        rows = []
        for c in CS.collections:
            for seg in c.get_paths():
                v = seg.vertices
                for (k, w) in v:
                    rows.append({"omega2": float(w), "Kphi": float(k)})
        plt.close(fig)
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            return True
    except Exception:
        pass

    # Fallback: outline of region >= level via filled contour
    try:
        fig, ax = plt.subplots()
        CF = ax.contourf(K, W, np.where(grid >= level, 1.0, np.nan), levels=[0.999,1.001])
        rows = []
        for c in CF.collections:
            for seg in c.get_paths():
                v = seg.vertices
                for (k, w) in v:
                    rows.append({"omega2": float(w), "Kphi": float(k)})
        plt.close(fig)
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            return True
    except Exception:
        pass

    return False

def main():
    args = parse_args()
    csv_path = args.csv
    outdir = args.outdir or os.path.dirname(csv_path)
    base = os.path.basename(csv_path)
    if base.endswith("_results.csv"):
        base = base[:-12]
    prefix = args.prefix or base
    os.makedirs(outdir, exist_ok=True)

    df, W, K, grid, status_grid = load_and_pivot(csv_path)

    counts = df["status"].value_counts().to_dict()
    finite_vals = grid[np.isfinite(grid)]
    vmin = float(np.nanmin(grid)) if finite_vals.size else np.nan
    vmax = float(np.nanmax(grid)) if finite_vals.size else np.nan
    q50 = float(np.nanquantile(finite_vals, 0.50)) if finite_vals.size else np.nan
    q90 = float(np.nanquantile(finite_vals, 0.90)) if finite_vals.size else np.nan

    best_loc = None
    if finite_vals.size:
        i,j = np.where(grid == vmax)
        if i.size:
            best_loc = {"omega2": float(W[i[0]]), "Kphi": float(K[j[0]]), "sync_index": float(vmax)}

    # Choose an actual contour level we'll use
    desired = args.threshold
    level = desired
    if finite_vals.size and not (vmin < desired < vmax):
        # Fall back to quantile inside range
        fallback = float(np.nanquantile(finite_vals, args.autoq))
        level = fallback
        warnings.warn(
            f"Requested threshold {desired:.4f} not in (min={vmin:.4f}, max={vmax:.4f}); "
            f"falling back to autoq={args.autoq:.2f} -> level={level:.4f}"
        )

    # Artifacts
    grid_csv = os.path.join(outdir, f"{prefix}_grid.csv")
    save_grid_csv(grid_csv, W, K, grid)

    heat_png = os.path.join(outdir, f"{prefix}_heatmap.png")
    plot_heatmap_with_boundary(heat_png, f"{prefix} — sync_index (level={level:.4f})", W, K, grid, level, dpi=args.dpi)

    to_png = os.path.join(outdir, f"{prefix}_timeouts.png")
    plot_timeout_mask_png(to_png, f"{prefix} — timeout/error mask", W, K, status_grid, dpi=args.dpi)

    boundary_csv = os.path.join(outdir, f"{prefix}_boundary_level{level:.4f}.csv")
    got_boundary = extract_boundary_csv(boundary_csv, W, K, grid, level)

    # Summary JSON
    best_json = os.path.join(outdir, f"{prefix}_best.json")
    with open(best_json, "w") as f:
        json.dump({
            "counts": counts,
            "stats": {"min": vmin, "max": vmax, "q50": q50, "q90": q90},
            "requested_threshold": desired,
            "used_level": level,
            "best": best_loc,
            "artifacts": {
                "grid_csv": grid_csv,
                "heatmap_png": heat_png,
                "timeouts_png": to_png,
                "boundary_csv": (boundary_csv if got_boundary else None)
            }
        }, f, indent=2)

    print("=== phase14_post summary ===")
    print("counts:", counts)
    print(f"min={vmin:.6f} max={vmax:.6f} q50={q50:.6f} q90={q90:.6f}")
    print("requested_threshold:", desired, "used_level:", level)
    print("best:", best_loc)
    print("grid_csv:", grid_csv)
    print("heatmap_png:", heat_png)
    print("timeouts_png:", to_png)
    print("boundary_csv:", boundary_csv if got_boundary else "(none)")
    print("best_json:", best_json)

if __name__ == "__main__":
    main()
