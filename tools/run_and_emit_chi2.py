#!/usr/bin/env python3
"""
Run joint_guard.py with whatever args you like, then emit a single summary line:
  chi2=... A=... f=... phi=... gamma=...
We *don’t* rely on filenames. We content-match JSONs written under outputs/**.

Usage:
  python3 tools/run_and_emit_chi2.py -- \
    --bao-csv ... --bao-cov ... --sn-csv ... --sn-cov ... \
    --H0 70.0 --Om 0.3 --Or 0.0 --Ok 0.0 --rd 147.1 \
    --prior-A-sigma 1.0 --k-params 5 --assume-per-rd --plus-lya \
    --gamma 1.740 --f 2.595
"""
import argparse, json, subprocess as sp, sys, glob, os, time
from datetime import datetime

def _mt(p):
    try: return os.path.getmtime(p)
    except: return 0.0

def _glob(patterns):
    out=[]
    for pat in patterns:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out))

def _extract(J):
    bp = J.get("best_params") or J.get("params") or {}
    def num(x):
        try: return float(x)
        except: return None
    chi2 = J.get("chi2") or J.get("chi_sq") or J.get("chisq") or J.get("chi2_min")
    return {
        "chi2": num(chi2),
        "A":    num(bp.get("A")),
        "f":    num(bp.get("f")),
        "phi":  num(bp.get("phi")),
        "gamma":num(bp.get("gamma")),
    }

def pick_json_by_content(g_target, f_target, since_ts):
    # 1) Prefer JSONs created since we started
    recent = [p for p in _glob(["outputs/**/*.json"]) if _mt(p) >= since_ts - 1.0]
    # 2) Otherwise, search everything (as fallback)
    pool = recent or _glob(["outputs/**/*.json"])
    cands=[]
    for p in pool:
        try:
            J=json.load(open(p))
        except Exception:
            continue
        rec=_extract(J)
        g,f=rec["gamma"], rec["f"]
        score=None
        if g is not None and f is not None:
            if abs(g - g_target) <= 1e-3 and abs(f - f_target) <= 1e-3:
                score=2  # strong match
            elif abs(g - g_target) <= 1e-2 and abs(f - f_target) <= 1e-2:
                score=1  # weak match
        if score is not None and rec["chi2"] is not None:
            cands.append((score, _mt(p), p, rec))
    if cands:
        cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return cands[0][2], cands[0][3]
    # No content match; take newest recent JSON with chi2 if any
    recent_with_chi = []
    for p in recent:
        try:
            J=json.load(open(p)); rec=_extract(J)
            if rec["chi2"] is not None:
                recent_with_chi.append(( _mt(p), p, rec ))
        except Exception:
            pass
    if recent_with_chi:
        recent_with_chi.sort(reverse=True)
        return recent_with_chi[0][1], recent_with_chi[0][2]
    return None, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--", dest="dashdash", help="separator", nargs="*")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="everything after -- is passed to joint_guard.py")
    args=ap.parse_args()

    # Extract gamma & f from the tail args for matching
    g_target = None; f_target = None
    tail = args.rest
    for i,a in enumerate(tail):
        if a == "--gamma" and i+1 < len(tail):
            try: g_target = float(tail[i+1])
            except: pass
        if a == "--f" and i+1 < len(tail):
            try: f_target = float(tail[i+1])
            except: pass

    run_started = time.time()
    cmd = ["python3", "joint_guard.py"] + tail
    proc = sp.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    # Try to find a JSON that matches these params (or the newest recent)
    if g_target is None or f_target is None:
        # still try to print something helpful
        path, rec = None, None
    else:
        path, rec = pick_json_by_content(g_target, f_target, run_started)

    if rec:
        print(f"chi2={rec['chi2']} A={rec['A']} f={rec['f']} phi={rec['phi']} gamma={rec['gamma']}")
        if path: print(f"[from] {path}")
        sys.exit(0)

    # Last resort: tell the user what changed recently
    recent = [(p, _mt(p)) for p in _glob(["outputs/**/*.json"]) if _mt(p) >= run_started - 2.0]
    if recent:
        recent.sort(key=lambda t: t[1], reverse=True)
        print("[WARN] No content-matched JSON. Recent JSONs after run:")
        for p,t in recent[:10]:
            ts = datetime.fromtimestamp(t).strftime("%H:%M:%S")
            print(f"  {ts}  {p}")
    else:
        print("[WARN] No JSON written after run start.")
    sys.exit(1)

if __name__ == "__main__":
    main()
