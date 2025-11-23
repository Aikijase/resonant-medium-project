#!/usr/bin/env python3
"""
Repair & compute A_gk_scale from growth (fσ8) data with zeros/percent-units.

Pipeline
--------
1) Load fs8 curves (CSV preferred): z, fs8_pred, fs8_fit   (*override names via flags*)
2) Fix units if model is in percent (median(model)/median(lcdm) >> 1): divide model by ~100
3) Replace non-positive values with NaN; **linearly interpolate** over NaNs on z-grid
4) Compute ratio r(z)=model/lcdm; drop non-finite; trim to keep_min..keep_max
5) Build **linear interpolator** for r(z); evaluate at g×κ bin z_eff with weights w=1/σ²
6) Weighted mean → A_gk_scale; optional clip to [clip_min, clip_max]
7) Save to outputs/phase2/growth_scale.json with rich diagnostics

Usage
-----
python3 tools/phase4_compute_agk_from_fs8_repair.py --model-col fs8_pred --lcdm-col fs8_fit --z-col z
"""
import os, sys, csv, json, math, statistics
from pathlib import Path

FS8_CSV   = "outputs/phase2/fs8_eval.csv"
FS8_JSON  = "outputs/phase2/fs8_eval.json"
BINS_CSV  = "data/isw/gk_bins.csv"
OUT_JSON  = "outputs/phase2/growth_scale.json"

MODEL_KEYS = ["fs8_pred","fs8_model","fs8_resonant","model","pred","resonant"]
LCDM_KEYS  = ["fs8_fit","fs8_lcdm","lcdm","baseline","ref"]
Z_KEYS     = ["z","z_eff","redshift"]

def read_csv_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = [{k.strip():v for k,v in row.items()} for row in r]
    for row in rows:
        for k,v in list(row.items()):
            try: row[k] = float(v)
            except Exception: pass
    return rows

def read_json_rows(path):
    J = json.load(open(path))
    if isinstance(J, dict):
        for k in ("points","data","rows"):
            if k in J and isinstance(J[k], list): return J[k]
    if isinstance(J, list): return J
    raise RuntimeError("Unrecognized fs8 JSON structure")

def pick_key(header, preferred, fallbacks):
    if preferred and preferred in header: return preferred
    for k in fallbacks:
        if k in header: return k
    raise RuntimeError(f"Column not found; header has: {sorted(header)}")

def summarize(name, arr):
    xs = [x for x in arr if isinstance(x,(int,float)) and math.isfinite(x)]
    if not xs: return f"{name}: no finite"
    return f"{name}: min={min(xs):.3f}  med={statistics.median(xs):.3f}  max={max(xs):.3f}"

def interp1d(xs, ys):
    pairs = sorted((x,y) for x,y in zip(xs,ys) if isinstance(x,(int,float)) and isinstance(y,(int,float)) and math.isfinite(x) and math.isfinite(y))
    if not pairs: raise RuntimeError("no finite points to interpolate")
    X=[p[0] for p in pairs]; Y=[p[1] for p in pairs]
    def f(x):
        if x <= X[0]: return Y[0]
        if x >= X[-1]: return Y[-1]
        lo,hi = 0, len(X)-1
        while hi-lo>1:
            mid=(lo+hi)//2
            if X[mid] <= x: lo=mid
            else: hi=mid
        x0,x1 = X[lo], X[hi]
        y0,y1 = Y[lo], Y[hi]
        t=(x-x0)/(x1-x0)
        return y0 + t*(y1-y0)
    return f, (X,Y)

def linear_interpolate_over_nans(zs, ys):
    """Return ys where NaNs are linearly interpolated on sorted z grid (ignores leading/trailing NaNs by carrying nearest)."""
    pairs = sorted(zip(zs,ys))
    z = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    # forward fill nearest finite for leading NaNs
    i=0
    while i < len(y) and not (isinstance(y[i], (int,float)) and math.isfinite(y[i])):
        i+=1
    if i < len(y):
        for k in range(i-1, -1, -1):
            y[k] = y[i]
    # backward fill nearest finite for trailing NaNs
    j=len(y)-1
    while j >=0 and not (isinstance(y[j], (int,float)) and math.isfinite(y[j])):
        j-=1
    if j >= 0:
        for k in range(j+1, len(y)):
            y[k] = y[j]
    # now interpolate interior runs of NaNs
    k=0
    while k < len(y):
        if isinstance(y[k], (int,float)) and math.isfinite(y[k]):
            k+=1; continue
        # start of NaN run
        start = k-1
        while k < len(y) and not (isinstance(y[k], (int,float)) and math.isfinite(y[k])):
            k+=1
        end = k
        if 0 <= start < len(y) and 0 <= end < len(y):
            z0, z1 = z[start], z[end]
            y0, y1 = y[start], y[end]
            for t, idx in enumerate(range(start+1, end), 1):
                frac = (z[idx]-z0)/(z1-z0) if z1!=z0 else 0.0
                y[idx] = y0 + frac*(y1 - y0)
    return z, y

def load_bins(path=BINS_CSV):
    out=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                out.append({"z": float(row["z_eff"]), "sigma": float(row["sigma_lit"])})
            except Exception:
                pass
    if not out:
        raise RuntimeError("No valid rows in gk_bins.csv")
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Repair & compute A_gk_scale from fs8 curves.")
    ap.add_argument("--model-col", default=None)
    ap.add_argument("--lcdm-col",  default=None)
    ap.add_argument("--z-col",     default=None)
    ap.add_argument("--keep-min",  type=float, default=0.6)
    ap.add_argument("--keep-max",  type=float, default=1.4)
    ap.add_argument("--clip-min",  type=float, default=0.85)
    ap.add_argument("--clip-max",  type=float, default=1.15)
    args = ap.parse_args()

    # load fs8
    if os.path.exists(FS8_CSV):
        rows = read_csv_rows(FS8_CSV); src = FS8_CSV
        print(f"[info] loaded {FS8_CSV}", file=sys.stderr)
    elif os.path.exists(FS8_JSON):
        rows = read_json_rows(FS8_JSON); src = FS8_JSON
        print(f"[info] loaded {FS8_JSON}", file=sys.stderr)
    else:
        print("[error] no fs8_eval.{csv,json} found", file=sys.stderr); sys.exit(2)
    header = set().union(*[{k for k in r.keys()} for r in rows])

    zkey = pick_key(header, args.z_col, Z_KEYS)
    mkey = pick_key(header, args.model_col, MODEL_KEYS)
    lkey = pick_key(header, args.lcdm_col , LCDM_KEYS)

    # extract raw vectors
    zs, mv, lv = [], [], []
    for r in rows:
        z = r.get(zkey); m = r.get(mkey); l = r.get(lkey)
        if isinstance(z,(int,float)) and isinstance(m,(int,float)) and isinstance(l,(int,float)):
            if math.isfinite(z) and math.isfinite(m) and math.isfinite(l):
                zs.append(float(z)); mv.append(float(m)); lv.append(float(l))

    print(f"[diag] raw {mkey}: {summarize(mkey, mv)}", file=sys.stderr)
    print(f"[diag] raw {lkey}: {summarize(lkey, lv)}", file=sys.stderr)

    # unit fix if model looks like percent vs fraction
    unit_scale = 1.0
    try:
        med_m = statistics.median([x for x in mv if math.isfinite(x) and x>0])
        med_l = statistics.median([x for x in lv if math.isfinite(x) and x>0])
        if med_m>0 and med_l>0 and med_m/med_l > 3.0:
            unit_scale = med_l/med_m
            mv = [x*unit_scale for x in mv]
            print(f"[info] unit-rescale model by {unit_scale:.3g} (med_m {med_m:.3f} → {med_m*unit_scale:.3f})", file=sys.stderr)
    except Exception:
        pass

    # replace non-positive with NaN then interpolate over NaNs
    mv2 = [x if (isinstance(x,(int,float)) and math.isfinite(x) and x>0) else float("nan") for x in mv]
    lv2 = [x if (isinstance(x,(int,float)) and math.isfinite(x) and x>0) else float("nan") for x in lv]
    z_sorted, mv3 = linear_interpolate_over_nans(zs, mv2)
    _,          lv3 = linear_interpolate_over_nans(zs, lv2)

    print(f"[diag] repaired {mkey}: {summarize(mkey, mv3)}", file=sys.stderr)
    print(f"[diag] repaired {lkey}: {summarize(lkey, lv3)}", file=sys.stderr)

    # ratios & trimming
    ratios = []
    for m,l in zip(mv3, lv3):
        if isinstance(m,(int,float)) and isinstance(l,(int,float)) and math.isfinite(m) and math.isfinite(l) and l>0:
            ratios.append(m/l)
        else:
            ratios.append(float("nan"))

    pairs = [(z,r) for z,r in zip(z_sorted, ratios) if math.isfinite(r)]
    kept  = [(z,r) for z,r in pairs if args.keep_min <= r <= args.keep_max]
    if not kept:
        # widen once, still guarded
        kept = [(z,r) for z,r in pairs if 0.5 <= r <= 1.5]
        print("[warn] widened keep window to [0.5, 1.5]", file=sys.stderr)
    if not kept:
        print("[error] no usable ratios after repair; writing neutral A_gk_scale=1.0", file=sys.stderr)
        A = 1.0
        meta = {"z_col": zkey, "model_col": mkey, "lcdm_col": lkey, "unit_scale": unit_scale, "repaired": True, "fallback": True}
        Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
        json.dump({"A_gk_scale": A, "source": "repair-fallback", "meta": meta}, open(OUT_JSON,"w"), indent=2)
        print(f"Wrote {OUT_JSON}  (A_gk_scale={A:.3f})")
        return

    z_keep = [z for z,_ in kept]
    r_keep = [r for _,r in kept]
    print(f"[diag] ratios kept: min={min(r_keep):.3f}  med={statistics.median(r_keep):.3f}  max={max(r_keep):.3f}", file=sys.stderr)

    # build ratio interpolator
    interp, _ = interp1d(z_keep, r_keep)

    # weight by gk bin sigmas
    bins = load_bins(BINS_CSV)
    num=0.0; den=0.0; log=[]
    for b in bins:
        r = float(interp(b["z"]))
        w = 1.0/(b["sigma"]**2) if b["sigma"]>0 else 0.0
        num += w*r; den += w
        log.append((b["z"], r, w))
    A = num/den if den>0 else statistics.median(r_keep)

    # final clipping
    clip_min, clip_max = float(args.clip_min), float(args.clip_max)
    A_clip = max(min(A, clip_max), clip_min)
    if A_clip != A:
        print(f"[info] clipped A_gk_scale from {A:.3f} to {A_clip:.3f} (bounds {clip_min}-{clip_max})", file=sys.stderr)

    meta = {
        "z_col": zkey, "model_col": mkey, "lcdm_col": lkey,
        "unit_scale": unit_scale, "repaired": True,
        "kept": len(kept), "total": len(pairs),
        "clip": [clip_min, clip_max]
    }

    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    json.dump({"A_gk_scale": A_clip, "source": f"{FS8_CSV} repaired", "meta": meta}, open(OUT_JSON,"w"), indent=2)
    print(f"Wrote {OUT_JSON}  (A_gk_scale={A_clip:.3f})")

if __name__ == "__main__":
    main()
