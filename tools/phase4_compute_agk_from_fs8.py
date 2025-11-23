#!/usr/bin/env python3
"""
Robust A_gk_scale from growth (fσ8) → outputs/phase2/growth_scale.json

Key features
- Explicit column selection (--model-col, --lcdm-col, --ratio-col, --z-col)
- Auto-detect common names (fs8_pred/fs8_fit supported)
- Cleans data: drop non-finite and <=0 values (invalid for ratios)
- Auto unit rescale: if model median >> LCDM median (e.g., ×100), rescales model
- Outlier guard: drop extreme ratios (> 3.0) before interpolation
- Sanity: if weighted mean ratio <0.6 or >1.4, try an invert and report
- Verbose diagnostics so you can see exactly what happened

Examples
--------
python3 tools/phase4_compute_agk_from_fs8.py --model-col fs8_pred --lcdm-col fs8_fit --z-col z
"""
import os, sys, csv, json, math, statistics
from pathlib import Path

BINS_CSV  = "data/isw/gk_bins.csv"
FS8_CSV   = "outputs/phase2/fs8_eval.csv"
FS8_JSON  = "outputs/phase2/fs8_eval.json"
OUT_JSON  = "outputs/phase2/growth_scale.json"

MODEL_KEYS = ["fs8_model","fs8_resonant","fs8_pred","model","pred","resonant"]
LCDM_KEYS  = ["fs8_lcdm","fs8_fit","lcdm","baseline","ref"]
RATIO_KEYS = ["scale","ratio","fs8_ratio","model_over_lcdm","resonant_over_lcdm"]
Z_KEYS     = ["z","z_eff","redshift"]

# --- IO helpers ---
def read_bins(path):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        need={"survey","bin_id","z_eff","A_lit","sigma_lit"}
        missing=need - set(r.fieldnames or [])
        if missing:
            raise RuntimeError(f"bins CSV missing columns: {sorted(missing)}")
        for i,row in enumerate(r,2):
            try:
                z=float(row["z_eff"]); s=float(row["sigma_lit"])
                if not math.isfinite(z) or not math.isfinite(s) or s<=0: raise ValueError
                rows.append({"survey":row["survey"],"bin_id":row["bin_id"],"z":z,"sigma":s})
            except Exception:
                print(f"[warn] skip bin line {i}", file=sys.stderr)
    if not rows: raise RuntimeError("no valid bins")
    return rows

def read_fs8_csv(path):
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        rows=[{k.strip():v for k,v in row.items()} for row in r]
    for row in rows:
        for k,v in list(row.items()):
            try: row[k]=float(v)
            except Exception: pass
    return rows

def read_fs8_json(path):
    J=json.load(open(path))
    if isinstance(J, dict):
        for k in ("points","data","rows"):
            if k in J and isinstance(J[k], list): return J[k]
    if isinstance(J, list): return J
    raise RuntimeError("Unrecognized fs8 JSON structure")

def pick_z_key(header, override=None):
    if override and override in header: return override
    for zk in Z_KEYS:
        if zk in header: return zk
    raise RuntimeError(f"No z-like column found. Header: {sorted(header)}")

def pick_columns(header, model_col=None, lcdm_col=None, ratio_col=None):
    if ratio_col:
        if ratio_col not in header: raise RuntimeError(f"--ratio-col '{ratio_col}' not in header")
        return ("ratio", ratio_col, None)
    if model_col and lcdm_col:
        if model_col not in header or lcdm_col not in header:
            raise RuntimeError(f"--model-col/--lcdm-col not in header ({model_col},{lcdm_col})")
        return ("pair", model_col, lcdm_col)
    for rk in RATIO_KEYS:
        if rk in header: return ("ratio", rk, None)
    mk = next((k for k in MODEL_KEYS if k in header), None)
    lk = next((k for k in LCDM_KEYS  if k in header), None)
    if mk and lk: return ("pair", mk, lk)
    raise RuntimeError(f"Could not find model/LCDM columns. Header: {sorted(header)}")

# --- math helpers ---
def build_interpolator(zs, ys):
    pairs=sorted((z,y) for z,y in zip(zs,ys)
                 if isinstance(z,(int,float)) and isinstance(y,(int,float))
                 and math.isfinite(z) and math.isfinite(y))
    if not pairs: raise RuntimeError("no finite points to interpolate")
    xs=[p[0] for p in pairs]; vs=[p[1] for p in pairs]
    def interp(x):
        if x<=xs[0]: return vs[0]
        if x>=xs[-1]: return vs[-1]
        lo,hi=0,len(xs)-1
        while hi-lo>1:
            mid=(lo+hi)//2
            if xs[mid]<=x: lo=mid
            else: hi=mid
        x0,x1=xs[lo],xs[hi]; y0,y1=vs[lo],vs[hi]
        t=(x - x0)/(x1 - x0)
        return y0 + t*(y1 - y0)
    return interp, (xs,vs)

def summarize(name, arr):
    xs=[x for x in arr if isinstance(x,(int,float)) and math.isfinite(x)]
    if not xs: return f"{name}: no finite"
    return f"{name}: min={min(xs):.3f}  med={statistics.median(xs):.3f}  max={max(xs):.3f}"

# --- core cleaning & ratio construction ---
def clean_positive(vals):
    return [x for x in vals if isinstance(x,(int,float)) and math.isfinite(x) and x>0]

def auto_unit_rescale(mvals, lvals):
    """Return scale factor s to multiply MODEL by so its median ~ LCDM median."""
    if not mvals or not lvals: return 1.0
    med_m = statistics.median(mvals)
    med_l = statistics.median(lvals)
    if med_m<=0 or med_l<=0: return 1.0
    # If model is in percent-ish units (e.g. ~50 vs 0.5), bring it down.
    ratio = med_m/med_l
    if ratio>3.0:
        s = med_l/med_m
        print(f"[info] unit-rescale: model*={s:.3g} (med_m={med_m:.3f}, med_l={med_l:.3f})", file=sys.stderr)
        return s
    return 1.0

def compute_ratio_series(fs8_rows, zkey, mode, A, B=None):
    zs=[]; ratio=[]
    if mode=="ratio":
        raw = [r.get(A) for r in fs8_rows]
        vals = clean_positive(raw)
        print(f"[diag] {summarize(A, vals)}", file=sys.stderr)
        for r in fs8_rows:
            z=r.get(zkey); rr=r.get(A)
            if isinstance(z,(int,float)) and isinstance(rr,(int,float)) and math.isfinite(z) and rr>0:
                zs.append(float(z)); ratio.append(float(rr))
        return zs, ratio, {"mode":"ratio","col":A,"unit_scale":1.0}
    # pair mode
    m_raw = [r.get(A) for r in fs8_rows]
    l_raw = [r.get(B) for r in fs8_rows]
    m_vals = clean_positive(m_raw)
    l_vals = clean_positive(l_raw)
    print(f"[diag] {summarize(A, m_vals)}", file=sys.stderr)
    print(f"[diag] {summarize(B, l_vals)}", file=sys.stderr)
    s = auto_unit_rescale(m_vals, l_vals)  # model *= s
    for r in fs8_rows:
        z=r.get(zkey); mm=r.get(A); ll=r.get(B)
        if isinstance(z,(int,float)) and isinstance(mm,(int,float)) and isinstance(ll,(int,float)):
            if math.isfinite(z) and mm>0 and ll>0:
                zs.append(float(z))
                ratio.append((mm*s)/ll)
    return zs, ratio, {"mode":"pair","model_col":A,"lcdm_col":B,"unit_scale":s}

def weighted_mean_over_bins(interp, bins):
    num=0.0; den=0.0; vals=[]
    for b in bins:
        r=float(interp(b["z"]))
        vals.append((b["z"], r))
        w=1.0/(b["sigma"]**2)
        num += w*r
        den += w
    return (num/den, vals)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compute A_gk_scale from fs8 growth outputs (robust).")
    ap.add_argument("--model-col", default=None)
    ap.add_argument("--lcdm-col",  default=None)
    ap.add_argument("--ratio-col", default=None)
    ap.add_argument("--z-col",     default=None)
    ap.add_argument("--bins",      default=BINS_CSV)
    ap.add_argument("--fs8csv",    default=FS8_CSV)
    ap.add_argument("--fs8json",   default=FS8_JSON)
    ap.add_argument("--out",       default=OUT_JSON)
    ap.add_argument("--invert",    action="store_true", help="invert the ratio (LCDM/model) after building it")
    args = ap.parse_args()

    # bins
    bins=read_bins(args.bins)

    # fs8 rows
    fs8_rows=None; src=None
    if os.path.exists(args.fs8csv):
        fs8_rows=read_fs8_csv(args.fs8csv); src=args.fs8csv
        print(f"[info] loaded fs8 CSV: {args.fs8csv}", file=sys.stderr)
    elif os.path.exists(args.fs8json):
        fs8_rows=read_fs8_json(args.fs8json); src=args.fs8json
        print(f"[info] loaded fs8 JSON: {args.fs8json}", file=sys.stderr)
    else:
        print("[error] no fs8_eval.{csv,json} found", file=sys.stderr); sys.exit(2)

    header=set().union(*[{k for k in r.keys()} for r in fs8_rows])
    zkey = pick_z_key(header, args.z_col)

    # prefer explicit; otherwise auto-detect, preferring (fs8_pred, fs8_fit)
    if args.ratio_col or (args.model_col and args.lcdm_col):
        mode, A, B = pick_columns(header, args.model_col, args.lcdm_col, args.ratio_col)
    else:
        if "fs8_pred" in header and "fs8_fit" in header:
            mode, A, B = "pair", "fs8_pred", "fs8_fit"
        else:
            mode, A, B = pick_columns(header, None, None, None)

    # build cleaned ratio series (with unit rescale if needed)
    zs, rr, meta = compute_ratio_series(fs8_rows, zkey, mode, A, B)

    # drop extreme ratios (likely artifacts) beyond 3σ-ish of unity band
    rr_filtered = []; zs_filtered=[]
    for z,r in zip(zs, rr):
        if math.isfinite(r) and 0.1 <= r <= 3.0:
            zs_filtered.append(z); rr_filtered.append(r)
    if len(rr_filtered) < max(5, len(rr)//2):
        print(f"[warn] many ratios were filtered ({len(rr_filtered)}/{len(rr)} kept). Check fs8 columns.", file=sys.stderr)
    print(f"[diag] ratio kept: min={min(rr_filtered):.3f}  med={statistics.median(rr_filtered):.3f}  max={max(rr_filtered):.3f}", file=sys.stderr)

    # build interpolator
    interp, (zx,rx) = build_interpolator(zs_filtered, rr_filtered)

    # first-pass weighted mean over bins
    A_scale, perbin = weighted_mean_over_bins(interp, bins)
    print(f"[diag] first pass A_gk_scale={A_scale:.3f}", file=sys.stderr)

    # optional invert
    if args.invert:
        A_scale = 1.0 / A_scale
        perbin = [(z, 1.0/r) for (z,r) in perbin]
        meta["inverted"] = True
        print(f"[info] manual invert → A_gk_scale={A_scale:.3f}", file=sys.stderr)
    else:
        # sanity auto-invert once if wildly off
        if A_scale < 0.6 or A_scale > 1.4:
            Ainvert = 1.0 / A_scale if A_scale!=0 else float("inf")
            if 0.6 <= Ainvert <= 1.4:
                A_scale = Ainvert
                perbin = [(z, 1.0/r) for (z,r) in perbin]
                meta["inverted"] = True
                print(f"[info] auto-invert → A_gk_scale={A_scale:.3f}", file=sys.stderr)

    # log per-bin ratios used
    for z, r in perbin:
        print(f"[bin] z={z:.3f}  ratio={r:.3f}", file=sys.stderr)

    # write out
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    out = {
        "A_gk_scale": A_scale,
        "source": f"{src} weighted by gk_bins sigmas",
        "meta": {"mode": meta.get("mode"), "model_col": meta.get("model_col"), "lcdm_col": meta.get("lcdm_col"),
                 "ratio_col": meta.get("col"), "unit_scale": meta.get("unit_scale"), "z_col": zkey,
                 "inverted": meta.get("inverted", False)}
    }
    with open(OUT_JSON,"w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}  (A_gk_scale={A_scale:.3f})")

if __name__ == "__main__":
    main()
