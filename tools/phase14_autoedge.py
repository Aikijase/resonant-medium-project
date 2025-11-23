#!/usr/bin/env python3
# tools/phase14_autoedge.py
# Auto-expand Phase14 zoom until a target contour is inside (min,max), then refine.
import argparse, os, sys, csv, json, subprocess, time
import numpy as np
import pandas as pd

PY = sys.executable

def run(cmd):
    p = subprocess.run(cmd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr

def stats_from_results(csv_path):
    df = pd.read_csv(csv_path)
    ok = df["status"].astype(str)=="ok"
    vals = pd.to_numeric(df.loc[ok,"sync_index"], errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size==0:
        return None
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "best": float(np.max(vals))
    }

def make_prefix(base, tag):
    return f"{base}_{tag}"

def zoom_once(args, span_w, span_k, step_w, step_k, prefix_tag, steps, timeout, retries, burn_in=300):
    prefix = make_prefix(args.prefix, prefix_tag)
    csv_path = os.path.join(args.outdir, f"{prefix}_results.csv")
    cmd = [
        PY, "tools/phase14_safe_zoom.py",
        "--center-omega2", str(args.center_omega2),
        "--center-Kphi",   str(args.center_Kphi),
        "--omega2-span",   str(span_w),
        "--Kphi-span",     str(span_k),
        "--omega2-step",   str(step_w),
        "--Kphi-step",     str(step_k),
        "--outdir",        args.outdir,
        "--prefix",        prefix,
        "--steps",         str(steps),
        "--burn_in",       str(burn_in),
        "--timeout",       str(timeout),
        "--retries",       str(retries)
    ]
    code, out, err = run(cmd)
    if code != 0:
        print(out)
        print(err, file=sys.stderr)
        raise SystemExit(f"safe_zoom failed ({code})")
    return csv_path, prefix

def post(csv_path, threshold, dpi=160):
    cmd = [PY, "tools/phase14_post.py", "--csv", csv_path, "--threshold", str(threshold), "--dpi", str(dpi)]
    code, out, err = run(cmd)
    print(out)
    if code != 0:
        print(err, file=sys.stderr)
        raise SystemExit(f"post failed ({code})")

def parse_args():
    ap = argparse.ArgumentParser(description="Auto-find Phase14 edge by expanding shells then refine.")
    ap.add_argument("--center-omega2", type=float, required=True)
    ap.add_argument("--center-Kphi",   type=float, required=True)
    ap.add_argument("--outdir",        type=str,   required=True)
    ap.add_argument("--prefix",        type=str,   default="p14_auto")
    ap.add_argument("--target",        type=float, default=0.95, help="desired sync_index level")
    ap.add_argument("--init-span",     type=float, default=0.12, help="initial span for omega2 and Kphi")
    ap.add_argument("--grow",          type=float, default=1.5,  help="span growth factor per shell")
    ap.add_argument("--max-span",      type=float, default=0.60, help="cap on span")
    ap.add_argument("--coarse-step",   type=float, default=0.01)
    ap.add_argument("--refine-step",   type=float, default=0.0025)
    ap.add_argument("--steps",         type=int,   default=10000, help="inner sim steps (coarse)")
    ap.add_argument("--steps-refine",  type=int,   default=14000, help="inner sim steps (refine)")
    ap.add_argument("--timeout",       type=float, default=300.0)
    ap.add_argument("--retries",       type=int,   default=1)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    span_w = span_k = args.init_span
    found = False
    iter_idx = 0

    # Shell expansion until target is inside (min,max)
    while True:
        iter_idx += 1
        tag = f"shell{iter_idx:02d}"
        print(f"[autoedge] pass {iter_idx}: span_w={span_w:.3f} span_k={span_k:.3f} step={args.coarse_step}")
        csv_path, coarse_prefix = zoom_once(args, span_w, span_k, args.coarse_step, args.coarse_step,
                                            tag, args.steps, args.timeout, args.retries)
        st = stats_from_results(csv_path)
        if st is None:
            raise SystemExit("No valid results; aborting.")
        print(f"[autoedge] stats: min={st['min']:.6f} max={st['max']:.6f} q50={st['q50']:.6f} q90={st['q90']:.6f}")

        if st["min"] < args.target < st["max"]:
            print(f"[autoedge] target {args.target} is inside range; proceeding to refine.")
            found = True
            post(csv_path, args.target)
            break

        if span_w >= args.max_span and span_k >= args.max_span:
            print(f"[autoedge] reached max span without bracketing target; will refine at outer shell anyway.")
            post(csv_path, st["q90"])  # still produce overlays
            break

        # grow shells
        span_w = min(args.max_span, span_w * args.grow)
        span_k = min(args.max_span, span_k * args.grow)

    # Refine pass if found
    if found:
        tag = "refine"
        print(f"[autoedge] refine pass with step={args.refine_step}")
        csv_path, refine_prefix = zoom_once(args, span_w, span_k, args.refine_step, args.refine_step,
                                            tag, args.steps_refine, args.timeout, args.retries)
        post(csv_path, args.target)
        print(f"[autoedge] done. refine csv: {csv_path}")
    else:
        print("[autoedge] done without bracketing; consider adjusting center or max-span.")

if __name__ == "__main__":
    raise SystemExit(main())
