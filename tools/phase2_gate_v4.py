#!/usr/bin/env python3
import argparse, json, os, math

def read_json(p):
    return json.load(open(p)) if (p and os.path.exists(p)) else None

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-json", required=True)  # e.g. .../kappa_v4_p1p0_fine.json
    ap.add_argument("--bao-alpha", type=float, default=None)
    ap.add_argument("--bao-alpha-file", default=None)  # optional JSON with {"alpha_hat": ...}
    ap.add_argument("--bao-tol", type=float, default=5e-4)
    ap.add_argument("--lensing-json", default="outputs/phase4/lensing_score.json")
    ap.add_argument("--spectral-json", default="outputs/phase8/phase8_metrics.json")
    args = ap.parse_args()

    S = read_json(args.suite_json)
    L = read_json(args.lensing_json)
    P8 = read_json(args.spectral_json)
    Aobj = read_json(args.bao_alpha_file) if args.bao_alpha_file else None

    ok = True; notes = []

    # 1) Information-criteria win?
    base = S["baseline"]; best = S["best"]
    win = (best["dAIC"] < 0) and (best["dBIC"] <= 0)
    if not win:
        ok = False; notes.append("No AIC/BIC win")

    # 2) BAO alpha guard
    alpha = args.bao_alpha
    if (alpha is None) and (Aobj is not None):
        alpha = Aobj.get("alpha_hat", 1.0)
    if alpha is not None:
        if abs(alpha - 1.0) > args.bao_tol:
            ok = False; notes.append(f"BAO α shift {alpha-1.0:+.2e} > tol {args.bao_tol:.1e}")

    # 3) Lensing & spectral sanity (presence or explicit PASS)
    if P8 is not None and ("verdict" in P8) and (str(P8["verdict"]).upper()!="PASS"):
        ok=False; notes.append("Phase-8 spectral not PASS")

    print("GATE:", "PASS" if ok else "FAIL")
    if notes: print("Notes:", "; ".join(notes))
