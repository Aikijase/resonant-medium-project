#!/usr/bin/env python3
"""
Phase-6 (v5): Resonant Response with z-window, gap repair, and *local* floor.

R(z) = L(z) / M(z)  (optionally times A_ratio * exp(-alpha * zeta))
But here we keep it simple/robust: just L/M with strong guards.

Guards:
- z-window: --z-min/--z-max
- gap repair (linear) with --max-gap
- floors:
    * optional absolute floor: --floor-frac * median(L)
    * optional percentile floor of positive M: --floor-percentile
    * **local floor**: eps * L(z) (default eps=0.02, adjustable by --eps)

Notes:
- No dependency on Phase-4/5 files (A_ratio/Gamma) to keep this stable.
- Plot is clipped to --clip-min/--clip-max for readability (data unaffected).
"""
import os, json, math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def interp_over_gaps(x, y, max_gap=None):
    """Linear repair of finite gaps; also carries short edge segments if <= max_gap."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    y2 = y.copy(); n_fix = n_left = 0; i = 0; N = len(y2)
    while i < N:
        if np.isfinite(y2[i]): i += 1; continue
        j = i
        while j < N and not np.isfinite(y2[j]): j += 1
        L = i - 1; R = j
        if L >= 0 and R < N and np.isfinite(y2[L]) and np.isfinite(y2[R]):
            gap = x[R] - x[L]
            if max_gap is None or gap <= max_gap:
                x0, x1 = x[L], x[R]; y0, y1 = y2[L], y2[R]
                for k in range(i, j):
                    t = (x[k] - x0) / (x1 - x0) if x1 != x0 else 0.0
                    y2[k] = y0 + t * (y1 - y0)
                n_fix += (j - i)
            else:
                n_left += (j - i)
        else:
            # edge segments
            if L < 0 and R < N and np.isfinite(y2[R]):
                if max_gap is None or (x[R] - x[0]) <= (max_gap or np.inf):
                    y2[i:j] = y2[R]; n_fix += (j - i)
                else:
                    n_left += (j - i)
            elif R >= N and L >= 0 and np.isfinite(y2[L]):
                if max_gap is None or (x[-1] - x[L]) <= (max_gap or np.inf):
                    y2[i:j] = y2[L]; n_fix += (j - i)
                else:
                    n_left += (j - i)
            else:
                n_left += (j - i)
        i = j
    return y2, int(n_fix), int(n_left)

def trimmed_stats(a, lo=0.1, hi=0.9):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not a.size:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    ql, qh = np.quantile(a, [lo, hi])
    s = a[(a >= ql) & (a <= qh)]
    return {"mean": float(np.mean(s)), "std": float(np.std(s, ddof=0)), "n": int(s.size)}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs8-csv", default="outputs/phase2/fs8_eval.csv")
    ap.add_argument("--z-col", default="z")
    ap.add_argument("--lcdm-col", default="fs8_fit")
    ap.add_argument("--model-col", default="fs8_pred")

    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=None)
    ap.add_argument("--max-gap", type=float, default=0.10)

    # floors
    ap.add_argument("--floor-frac", type=float, default=0.0, help="abs floor = frac * median(L)")
    ap.add_argument("--floor-percentile", type=float, default=None, help="p-th percentile of positive M")
    ap.add_argument("--eps", type=float, default=0.02, help="local floor M(z) >= eps * L(z)")

    # plot & outputs
    ap.add_argument("--clip-min", type=float, default=0.6)
    ap.add_argument("--clip-max", type=float, default=1.4)
    ap.add_argument("--out-csv", default="outputs/phase6/resonant_response.csv")
    ap.add_argument("--out-json", default="outputs/phase6/resonant_response.json")
    ap.add_argument("--out-png", default="plots/phase6/resonant_response.png")
    args = ap.parse_args()

    Path("outputs/phase6").mkdir(parents=True, exist_ok=True)
    Path("plots/phase6").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.fs8_csv)
    for c in (args.z_col, args.lcdm_col, args.model_col):
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} in {args.fs8_csv}")

    z = pd.to_numeric(df[args.z_col], errors="coerce").to_numpy()
    L = pd.to_numeric(df[args.lcdm_col], errors="coerce").to_numpy()
    M = pd.to_numeric(df[args.model_col], errors="coerce").to_numpy()

    # z-window
    mask = np.isfinite(z)
    if args.z_min is not None: mask &= (z >= args.z_min)
    if args.z_max is not None: mask &= (z <= args.z_max)
    z, L, M = z[mask], L[mask], M[mask]

    # sanitize
    L = np.where((~np.isfinite(L)) | (L <= 0), np.nan, L)
    M = np.where((~np.isfinite(M)) | (M <= 0), np.nan, M)

    # gap repair
    L_rep, L_fix, L_left = interp_over_gaps(z, L, args.max_gap)
    M_rep, M_fix, M_left = interp_over_gaps(z, M, args.max_gap)

    # floors
    medL = np.nanmedian(L_rep)
    floor_val = 0.0
    if args.floor_frac > 0 and np.isfinite(medL):
        floor_val = max(floor_val, args.floor_frac * medL)
    if args.floor_percentile is not None:
        posM = M_rep[np.isfinite(M_rep) & (M_rep > 0)]
        if posM.size:
            floor_val = max(floor_val, float(np.percentile(posM, args.floor_percentile)))

    # local floor: eps * L(z)
    eps = float(args.eps)
    local_floor = eps * L_rep
    if floor_val > 0:
        local_floor = np.maximum(local_floor, floor_val)

    M_rep = np.where(np.isfinite(M_rep), np.maximum(M_rep, local_floor), np.nan)

    # compute R
    R = np.divide(L_rep, M_rep)
    finite = np.isfinite(R)
    tstats = trimmed_stats(R[finite], 0.1, 0.9)

    # save CSV
    out = df.loc[mask].copy()
    out["fs8_lcdm_repaired"] = L_rep
    out["fs8_model_repaired"] = M_rep
    out["R_res"] = R
    out.to_csv(args.out_csv, index=False)

    # save JSON
    meta = {
        "z_window": [args.z_min, args.z_max],
        "repairs": {"lcdm_fixed": L_fix, "lcdm_left": L_left, "model_fixed": M_fix, "model_left": M_left},
        "floor": {"eps": eps, "floor_frac": args.floor_frac, "floor_percentile": args.floor_percentile,
                  "applied_abs_floor": float(np.nanmax(local_floor)) if np.isfinite(local_floor).any() else 0.0},
        "stats_trimmed_10_90": tstats
    }
    with open(args.out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.out_csv} and {args.out_json}")

    # plot (view-only clipping)
    m = finite & (R > args.clip_min) & (R < args.clip_max)
    plt.figure(figsize=(6.4, 4.2))
    plt.axhline(1.0, ls="--", lw=1)
    if m.any():
        plt.plot(out[args.z_col].to_numpy()[m], R[m], "o-", label="R(z)")
    else:
        plt.plot([], [], "o-", label="R(z)")
    plt.xlabel("Redshift z"); plt.ylabel("Resonant Response R(z)")
    plt.title("Phase-6: Resonant Response (v5, local floor)")
    plt.grid(alpha=.3); plt.legend()
    Path(os.path.dirname(args.out_png)).mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out_png, dpi=160)
    print(f"Wrote {args.out_png}")

if __name__ == "__main__":
    main()
