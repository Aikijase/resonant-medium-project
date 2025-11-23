#!/usr/bin/env python3
"""
Phase 14 — Contour width & area between two Si levels.

Outputs:
- Contour PNG + CSVs for each level
- Width-vs-omega2 PNG + CSV (envelope method)
- Console stats
- Two area metrics:
    A_trapz : ∫ [Kphi_max(high) - Kphi_min(low)] d(omega2)  (envelope-based)
    A_mask  : ∬ 1_{low<=Si<=high} d(omega2) dKphi         (grid mask between levels)
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ---------- helpers ----------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_shell(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["omega2", "Kphi", "sync_index"])
    if df.empty:
        print("ERROR: shell CSV has no valid rows after dropping NaNs.", file=sys.stderr)
        sys.exit(1)
    return df

def make_grid(df: pd.DataFrame, nw=400, nk=400, pad=0.02):
    w = df["omega2"].to_numpy(); k = df["Kphi"].to_numpy()
    wmin, wmax = float(w.min()), float(w.max())
    kmin, kmax = float(k.min()), float(k.max())
    wr = wmax - wmin; kr = kmax - kmin
    w_lin = np.linspace(wmin - pad*wr, wmax + pad*wr, nw)
    k_lin = np.linspace(kmin - pad*kr, kmax + pad*kr, nk)
    W, K = np.meshgrid(w_lin, k_lin)
    return W, K

def interp_to_grid(df: pd.DataFrame, W, K):
    w = df["omega2"].to_numpy(); k = df["Kphi"].to_numpy(); z = df["sync_index"].to_numpy()
    Z_lin = griddata((w, k), z, (W, K), method="linear")
    Z_near = griddata((w, k), z, (W, K), method="nearest")
    Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)
    return Z

def save_contours(W, K, Z, levels, outdir, prefix):
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=160)
    cs = ax.contour(W, K, Z, levels=levels)
    ax.clabel(cs, cs.levels, inline=True, fontsize=8)
    ax.set_xlabel(r'$\omega^2$'); ax.set_ylabel(r'$K_\phi$')
    ax.set_title('Sync index contours')
    fig.tight_layout()
    contour_png = os.path.join(outdir, f"{prefix}_contours.png")
    fig.savefig(contour_png); plt.close(fig)

    paths = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        rows = []
        for seg_id, seg in enumerate(segs):
            for (w, k) in seg:
                rows.append((lev, seg_id, float(w), float(k)))
        df = pd.DataFrame(rows, columns=["level","segment_id","omega2","Kphi"])
        csv = os.path.join(outdir, f"{prefix}_contour_si{lev:0.3f}.csv")
        df.to_csv(csv, index=False)
        paths.append(csv)
    return paths, contour_png

def envelope_from_contour(csv_path: str, wbins=600) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna()
    if df.empty:
        return pd.DataFrame(columns=["wbin","omega2","Kphi_min","Kphi_max"])
    wmin, wmax = float(df["omega2"].min()), float(df["omega2"].max())
    if wmin == wmax:
        df["wbin"] = 1
    else:
        bins = np.linspace(wmin, wmax, wbins)
        wbin = np.clip(np.digitize(df["omega2"].to_numpy(), bins), 1, len(bins)-1)
        df = df.assign(wbin=wbin)
    env = (df.groupby("wbin")
             .agg(omega2=("omega2","mean"),
                  Kphi_min=("Kphi","min"),
                  Kphi_max=("Kphi","max"))
             .reset_index()
             .dropna())
    return env

def merge_envelopes_strict(lo_env: pd.DataFrame, hi_env: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(lo_env, hi_env, on="wbin", suffixes=("_lo","_hi"))
    if m.empty: return m
    m["width"] = m["Kphi_max_hi"] - m["Kphi_min_lo"]
    m["omega2"] = 0.5*(m["omega2_lo"] + m["omega2_hi"])
    m = m.dropna(subset=["width","omega2"]).sort_values("omega2")
    return m

def trapz_area(df: pd.DataFrame) -> float:
    if df.empty: return float("nan")
    return float(np.trapz(df["width"].to_numpy(), df["omega2"].to_numpy()))

def plot_width(df: pd.DataFrame, outdir: str, prefix: str, low: float, high: float):
    if df.empty: return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.plot(df["omega2"].to_numpy(), df["width"].to_numpy(), lw=1.6)
    ax.set_xlabel(r'$\omega^2$')
    ax.set_ylabel(r'Width in $K_\phi$ (outer %.2f − inner %.2f)' % (high, low))
    ax.set_title('Lock-tongue width vs $\omega^2$')
    fig.tight_layout()
    png = os.path.join(outdir, f"{prefix}_width_si{low:0.3f}_to_si{high:0.3f}.png")
    fig.savefig(png); plt.close(fig)
    return png

def area_mask_between_levels(W, K, Z, low, high):
    """Area of the band {low <= Z <= high} via grid-cell integration."""
    # mask of interior cells where Z is between levels
    mask = (Z >= low) & (Z <= high)
    # approximate cell sizes
    dw = np.diff(W[0,:]).mean()
    dk = np.diff(K[:,0]).mean()
    # sum over mask
    area = float(mask.sum()) * dw * dk
    return area, mask

def plot_mask(W, K, mask, outdir, prefix, low, high):
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=160)
    ax.imshow(mask[::-1,:], extent=[W.min(), W.max(), K.min(), K.max()], aspect='auto')
    ax.set_xlabel(r'$\omega^2$'); ax.set_ylabel(r'$K_\phi$')
    ax.set_title(f'Band mask: {low:.2f} ≤ Si ≤ {high:.2f}')
    fig.tight_layout()
    png = os.path.join(outdir, f"{prefix}_mask_si{low:0.3f}_to_si{high:0.3f}.png")
    fig.savefig(png); plt.close(fig)
    return png

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Phase14: contour width & area between two Si levels.")
    ap.add_argument("--shell-csv", required=True, help="CSV with columns: omega2,Kphi,sync_index")
    ap.add_argument("--outdir", default="outputs/phase14")
    ap.add_argument("--prefix", default="p14_edge")
    ap.add_argument("--levels", default="0.90,0.92,0.94,0.96,0.98",
                    help="Comma list of Si levels for contouring")
    ap.add_argument("--low", type=float, default=0.90, help="Inner (lower) Si level")
    ap.add_argument("--high", type=float, default=0.98, help="Outer (higher) Si level")
    ap.add_argument("--nw", type=int, default=400)
    ap.add_argument("--nk", type=int, default=400)
    ap.add_argument("--wbins", type=int, default=600)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = load_shell(args.shell_csv)
    W, K = make_grid(df, args.nw, args.nk)
    Z = interp_to_grid(df, W, K)

    # contours + CSVs
    levels = [float(x.strip()) for x in args.levels.split(",")]
    contour_csvs, contour_png = save_contours(W, K, Z, levels, args.outdir, args.prefix)

    # envelopes at low/high
    by_level = {float(os.path.basename(p).split("si")[-1].replace(".csv","")): p
                for p in contour_csvs}
    if args.low not in by_level or args.high not in by_level:
        print("Requested low/high levels missing. Found:", sorted(by_level), file=sys.stderr)
        sys.exit(2)

    lo_env = envelope_from_contour(by_level[args.low], args.wbins)
    hi_env = envelope_from_contour(by_level[args.high], args.wbins)
    merged = merge_envelopes_strict(lo_env, hi_env)

    if merged.empty:
        print("No overlapping bins between low/high envelopes. Densify sampling and re-run.",
              file=sys.stderr)
        sys.exit(3)

    # width stats + trapz area
    stats = merged["width"].describe().to_dict()
    area_trapz = trapz_area(merged)

    # mask area (grid)
    area_mask, mask = area_mask_between_levels(W, K, Z, args.low, args.high)

    # save width CSV/PNG
    width_csv = os.path.join(args.outdir,
        f"{args.prefix}_width_envelope_si{args.low:0.3f}_to_si{args.high:0.3f}.csv")
    merged[["omega2","width"]].to_csv(width_csv, index=False)
    width_png = plot_width(merged, args.outdir, args.prefix, args.low, args.high)

    # save mask PNG
    mask_png = plot_mask(W, K, mask, args.outdir, args.prefix, args.low, args.high)

    # console summary
    print(f"contour_png: {contour_png}")
    print(f"contour_csvs: {sorted(contour_csvs)}")
    print(f"width_envelope_csv: {width_csv}")
    print(f"width_png: {width_png}")
    print(f"mask_png: {mask_png}")
    print("width stats:", {k: float(v) for k,v in stats.items()})
    print(f"A_trapz (envelope)  : {area_trapz:.12f}  [Kphi·ω²]")
    print(f"A_mask  (grid band) : {area_mask:.12f}  [Kphi·ω²]")
    print("omega2_span_used (trapz):", 
          float(merged["omega2"].min()), "→", float(merged["omega2"].max()))

if __name__ == "__main__":
    raise SystemExit(main())
