#!/usr/bin/env python3
"""
Phase-6 (v3): Robust Resonant Response R(z) with gap repair and safe flooring.

R(z) = (fs8_LCDM / fs8_model) * (A_hat / A_pred) * exp(-alpha * zeta(z))

Safeties:
- Treat non-finite or <=0 values as missing; linear-interpolate over *short* gaps
- Optionally floor model to `floor_frac * median(fs8_LCDM)` after interpolation
- Drop any remaining non-finite or extreme points before stats/plot
- Provide both finite-only stats and 10–90% trimmed stats

CLI example:
  python3 tools/phase6_resonant_response_v3.py \
    --fs8-csv outputs/phase2/fs8_eval.csv \
    --z-col z --lcdm-col fs8_fit --model-col fs8_pred \
    --alpha 0.0 --max-gap 0.25 --floor-frac 0.05
"""
import os, json, math, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def load_json(path, default=None):
    try:
        with open(path,"r") as f: return json.load(f)
    except Exception:
        return default

def interp_over_gaps(x, y, max_gap=None):
    """
    Interpolate y over NaNs on x-grid.
    If a contiguous NaN span in x exceeds max_gap, leave it NaN.
    x must be 1D increasing.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    isn = ~np.isfinite(y)
    if not isn.any():
        return y.copy(), 0, 0

    y2 = y.copy()
    # locate runs of NaNs
    n_repaired = 0
    n_left = 0
    i = 0
    N = len(y2)
    while i < N:
        if not np.isnan(y2[i]):
            i += 1
            continue
        # start of NaN run
        j = i
        while j < N and np.isnan(y2[j]):
            j += 1
        # run is [i, j-1]
        left = i-1
        right = j
        if left >= 0 and right < N and np.isfinite(y2[left]) and np.isfinite(y2[right]):
            gap = x[right] - x[left]
            if (max_gap is None) or (gap <= max_gap):
                # linear interpolate
                x0, x1 = x[left], x[right]
                y0, y1 = y2[left], y2[right]
                for k in range(i, j):
                    t = (x[k]-x0)/(x1-x0) if x1!=x0 else 0.0
                    y2[k] = y0 + t*(y1 - y0)
                n_repaired += (j - i)
            else:
                # leave NaNs
                n_left += (j - i)
        else:
            # Leading/trailing NaNs: carry nearest finite if span <= max_gap, else leave
            if left < 0 and right < N and np.isfinite(y2[right]):
                # treat as short if length in x <= max_gap
                if (max_gap is None) or ((x[right] - x[0]) <= max_gap):
                    y2[i:j] = y2[right]
                    n_repaired += (j - i)
                else:
                    n_left += (j - i)
            elif right >= N and left >=0 and np.isfinite(y2[left]):
                if (max_gap is None) or ((x[-1] - x[left]) <= max_gap):
                    y2[i:j] = y2[left]
                    n_repaired += (j - i)
                else:
                    n_left += (j - i)
            else:
                n_left += (j - i)
        i = j
    return y2, n_repaired, n_left

def trimmed_stats(arr, lo=0.1, hi=0.9):
    x = np.asarray(arr, float)
    x = x[np.isfinite(x)]
    if x.size == 0: return {"mean": float("nan"), "std": float("nan"), "n": 0}
    qlo, qhi = np.quantile(x, [lo, hi])
    sel = (x >= qlo) & (x <= qhi)
    xs = x[sel]
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=0)), "n": int(xs.size)}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs8-csv", default="outputs/phase2/fs8_eval.csv")
    ap.add_argument("--z-col", default="z")
    ap.add_argument("--lcdm-col", default="fs8_fit")
    ap.add_argument("--model-col", default="fs8_pred")
    ap.add_argument("--gk-multi", default="outputs/phase4/gk_multi.json")
    ap.add_argument("--phase5-runs", default="outputs/phase5/damping_runs.json")
    ap.add_argument("--alpha", type=float, default=0.0, help="damping strength multiplier on zeta(z)")
    ap.add_argument("--max-gap", type=float, default=0.25, help="max ∆z span to interpolate across; larger gaps left NaN")
    ap.add_argument("--floor-frac", type=float, default=0.05, help="floor = this * median(fs8_LCDM) applied to model after interpolation")
    ap.add_argument("--clip-min", type=float, default=0.6)
    ap.add_argument("--clip-max", type=float, default=1.4)
    ap.add_argument("--out-csv", default="outputs/phase6/resonant_response.csv")
    ap.add_argument("--out-json", default="outputs/phase6/resonant_response.json")
    ap.add_argument("--out-png", default="plots/phase6/resonant_response.png")
    args = ap.parse_args()

    Path("outputs/phase6").mkdir(parents=True, exist_ok=True)
    Path("plots/phase6").mkdir(parents=True, exist_ok=True)

    # Load fs8
    df = pd.read_csv(args.fs8_csv)
    for c in (args.z_col, args.lcdm_col, args.model_col):
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in {args.fs8_csv}. Columns: {list(df.columns)}")

    z = pd.to_numeric(df[args.z_col], errors="coerce").to_numpy()
    L = pd.to_numeric(df[args.lcdm_col], errors="coerce").to_numpy()
    M = pd.to_numeric(df[args.model_col], errors="coerce").to_numpy()

    # mark invalids as NaN
    L = np.where((~np.isfinite(L)) | (L<=0), np.nan, L)
    M_raw = np.where((~np.isfinite(M)) | (M<=0), np.nan, M)

    # unit rescue: if model median >> LCDM median, rescale model down
    def finite_pos(x): 
        x = x[np.isfinite(x) & (x>0)]
        return x
    med_L = np.median(finite_pos(L)) if finite_pos(L).size else np.nan
    med_M = np.median(finite_pos(M_raw)) if finite_pos(M_raw).size else np.nan
    unit_scale = 1.0
    if np.isfinite(med_L) and np.isfinite(med_M) and med_M/med_L > 3.0:
        unit_scale = med_L/med_M
    M_scaled = M_raw * unit_scale

    # repair gaps in L and M (short gaps only)
    L_rep, L_fix, L_left = interp_over_gaps(z, L, max_gap=args.max_gap)
    M_rep, M_fix, M_left = interp_over_gaps(z, M_scaled, max_gap=args.max_gap)

    # apply a gentle floor to model after interpolation
    floor_val = (args.floor_frac * med_L) if np.isfinite(med_L) else 0.0
    if floor_val > 0:
        M_rep = np.where(np.isfinite(M_rep), np.maximum(M_rep, floor_val), np.nan)

    # ratio and R
    ratio = np.divide(M_rep, L_rep)
    R = np.divide(1.0, ratio)  # L/M
    # Phase-4 A_ratio
    gk = load_json(args.gk_multi, {})
    A_pred = gk.get("scale_model", 1.0) or 1.0
    A_hat  = gk.get("A_hat", 1.0)
    A_ratio = (A_hat / A_pred) if A_pred != 0 else 1.0

    # Phase-5 damping → zeta(z) mapped across z by index (no redshift model here)
    runs = load_json(args.phase5_runs, {"runs":[]}).get("runs", [])
    if runs:
        zeta = np.array([r.get("zeta", 0.0) for r in runs], dtype=float)
        idx_r = np.linspace(0, 1, len(zeta))
        idx_z = np.linspace(0, 1, len(z))
        ZETA  = np.interp(idx_z, idx_r, zeta)
    else:
        ZETA = np.zeros_like(z)
    Gamma = args.alpha * ZETA
    R = R * A_ratio * np.exp(-Gamma)

    # mask non-finite results
    finite = np.isfinite(R)
    Rf = R[finite]
    # basic stats
    mean_R = float(Rf.mean()) if Rf.size else float("nan")
    std_R  = float(Rf.std(ddof=0)) if Rf.size else float("nan")
    # trimmed stats
    tstats = trimmed_stats(Rf, 0.1, 0.9)

    # save CSV
    df_out = df.copy()
    df_out["fs8_lcdm_repaired"] = L_rep
    df_out["fs8_model_repaired"] = M_rep
    df_out["R_res"]   = R
    df_out["Gamma"]   = Gamma
    df_out["A_ratio"] = A_ratio
    df_out["unit_scale_model"] = unit_scale
    df_out.to_csv(args.out_csv, index=False)

    # save JSON
    meta = {
        "alpha": args.alpha,
        "max_gap": args.max_gap,
        "floor_frac": args.floor_frac,
        "clip_band": [args.clip_min, args.clip_max],
        "unit_scale_model": unit_scale,
        "repairs": {
            "lcdm_fixed": int(L_fix), "lcdm_left": int(L_left),
            "model_fixed": int(M_fix), "model_left": int(M_left),
        },
        "A_ratio": A_ratio,
        "stats_finite": {"mean": mean_R, "std": std_R, "n": int(Rf.size)},
        "stats_trimmed_10_90": tstats
    }
    with open(args.out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.out_csv} and {args.out_json}")

    # plot (clip for axis sanity; do not alter CSV values)
    mask = finite & (R > args.clip_min) & (R < args.clip_max)
    plt.figure(figsize=(6.4,4.2))
    plt.axhline(1.0, ls="--", lw=1)
    if mask.any():
        plt.plot(z[mask], R[mask], "o-", label="R(z)")
    else:
        plt.plot([], [], label="R(z) (no points in clip band)")
    plt.xlabel("Redshift z"); plt.ylabel("Resonant Response R(z)")
    plt.title("Phase-6: Resonant Response (robust)")
    plt.grid(alpha=0.3); plt.legend()
    Path(os.path.dirname(args.out_png)).mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out_png, dpi=160)
    print(f"Wrote {args.out_png}")

if __name__ == "__main__":
    main()
