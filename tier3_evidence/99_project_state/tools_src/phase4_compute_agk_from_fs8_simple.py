#!/usr/bin/env python3
"""
Compute A_gk_scale robustly from fs8 curves by median-of-ratios with trimming.

Inputs
------
- outputs/phase2/fs8_eval.csv (preferred) or fs8_eval.json
  Expected columns (auto-detected unless overridden):
    z column: one of [z, z_eff, redshift] or --z-col
    model: one of [fs8_pred, fs8_model, fs8_resonant, model, pred, resonant] or --model-col
    lcdm : one of [fs8_fit, fs8_lcdm, lcdm, baseline, ref] or --lcdm-col

Method
------
1) Build ratios r(z) = model(z) / lcdm(z) on rows where both > 0 and finite.
2) Optional unit rescue: if median(model) >> median(lcdm) (e.g., ~×100), rescale model.
3) Trim outliers outside [0.6, 1.4] (configurable via --keep-min/--keep-max).
4) Weighted median with weights w(z) ~ (1 + z)  (softly up-weights mid/high-z bins),
   but any zeros/NaNs are excluded.
5) Clip final A_gk_scale to [clip_min, clip_max] (default [0.8, 1.2]) to be safe.
6) Write outputs/phase2/growth_scale.json: {"A_gk_scale": <float>, meta...}

Usage
-----
# simplest (auto-detect columns)
python3 tools/phase4_compute_agk_from_fs8_simple.py

# explicit columns
python3 tools/phase4_compute_agk_from_fs8_simple.py --model-col fs8_pred --lcdm-col fs8_fit --z-col z

# custom bands
python3 tools/phase4_compute_agk_from_fs8_simple.py --keep-min 0.7 --keep-max 1.3 --clip-min 0.85 --clip-max 1.15
"""
import os, sys, csv, json, math, statistics
from pathlib import Path

FS8_CSV   = "outputs/phase2/fs8_eval.csv"
FS8_JSON  = "outputs/phase2/fs8_eval.json"
OUT_JSON  = "outputs/phase2/growth_scale.json"
BINS_CSV  = "data/isw/gk_bins.csv"  # only used for logging z-range

MODEL_KEYS = ["fs8_pred","fs8_model","fs8_resonant","model","pred","resonant"]
LCDM_KEYS  = ["fs8_fit","fs8_lcdm","lcdm","baseline","ref"]
Z_KEYS     = ["z","z_eff","redshift"]

def read_csv_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = [{k.strip():v for k,v in row.items()} for row in r]
    # try to cast
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
    raise RuntimeError(f"Columns not found; header has: {sorted(header)}")

def weighted_median(values, weights):
    # both lists finite, positive weights
    pairs = sorted((v,w) for v,w in zip(values,weights) if math.isfinite(v) and math.isfinite(w) and w>0)
    if not pairs:
        raise RuntimeError("No valid data for weighted median")
    total = sum(w for _,w in pairs)
    acc = 0.0
    for v,w in pairs:
        acc += w
        if acc >= 0.5*total:
            return v
    return pairs[-1][0]

def summarize(name, arr):
    xs = [x for x in arr if isinstance(x,(int,float)) and math.isfinite(x)]
    if not xs: return f"{name}: no finite"
    return f"{name}: min={min(xs):.3f}  med={statistics.median(xs):.3f}  max={max(xs):.3f}"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-col", default=None)
    ap.add_argument("--lcdm-col",  default=None)
    ap.add_argument("--z-col",     default=None)
    ap.add_argument("--keep-min",  type=float, default=0.6)
    ap.add_argument("--keep-max",  type=float, default=1.4)
    ap.add_argument("--clip-min",  type=float, default=0.8)
    ap.add_argument("--clip-max",  type=float, default=1.2)
    args = ap.parse_args()

    # load curves
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

    # collect finite, positive pairs
    zs, mv, lv = [], [], []
    for r in rows:
        z = r.get(zkey); m = r.get(mkey); l = r.get(lkey)
        if isinstance(z,(int,float)) and isinstance(m,(int,float)) and isinstance(l,(int,float)):
            if math.isfinite(z) and math.isfinite(m) and math.isfinite(l) and m>0 and l>0:
                zs.append(float(z)); mv.append(float(m)); lv.append(float(l))

    if not zs:
        print("[error] no finite positive pairs in fs8 table", file=sys.stderr); sys.exit(2)

    print(f"[diag] {summarize(mkey, mv)}", file=sys.stderr)
    print(f"[diag] {summarize(lkey, lv)}", file=sys.stderr)

    # crude unit rescue (percent-like → fraction)
    med_m = statistics.median(mv); med_l = statistics.median(lv)
    unit_scale = 1.0
    if med_m>0 and med_l>0:
        ratio_med = med_m/med_l
        if ratio_med > 3.0:
            unit_scale = med_l/med_m
            mv = [x*unit_scale for x in mv]
            print(f"[info] unit-rescale model by {unit_scale:.3g} to match LCDM scale", file=sys.stderr)

    # ratios & trimming
    ratios = [m/l for m,l in zip(mv,lv) if l>0]
    kept = [(z,r) for z,r in zip(zs,ratios) if math.isfinite(r) and args.keep_min <= r <= args.keep_max]
    if len(kept) < max(5, len(ratios)//3):
        print(f"[warn] few ratios kept ({len(kept)}/{len(ratios)}). Check fs8 columns.", file=sys.stderr)

    if not kept:
        print("[error] no ratios survived trimming", file=sys.stderr); sys.exit(2)

    z_keep = [z for z,_ in kept]
    r_keep = [r for _,r in kept]
    # gentle redshift weight (1+z); avoids relying on gk_bins here
    w_keep = [1.0+z for z in z_keep]

    A_med = weighted_median(r_keep, w_keep)
    A_clip = max(min(A_med, args.clip_max), args.clip_min)
    clipped = (A_clip != A_med)

    print(f"[diag] ratios kept: {summarize('r', r_keep)}", file=sys.stderr)
    print(f"[diag] weighted median = {A_med:.3f}", file=sys.stderr)
    if clipped:
        print(f"[info] clipped to [{args.clip_min:.2f}, {args.clip_max:.2f}] → A_gk_scale={A_clip:.3f}", file=sys.stderr)

    # write out
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    out = {
        "A_gk_scale": A_clip,
        "source": f"{src} weighted median of ratios (trim={args.keep_min}-{args.keep_max}, clip={args.clip_min}-{args.clip_max})",
        "meta": {
            "z_col": zkey, "model_col": mkey, "lcdm_col": lkey,
            "unit_scale": unit_scale, "kept": len(kept), "total": len(ratios),
            "r_median_uncut": A_med
        }
    }
    with open(OUT_JSON,"w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}  (A_gk_scale={A_clip:.3f})")

if __name__ == "__main__":
    main()
