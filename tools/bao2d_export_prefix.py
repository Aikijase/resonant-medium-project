# tools/bao2d_export_prefix.py
import json, subprocess as sp, numpy as np, pathlib, csv, glob, time, os, sys
from datetime import datetime

OUTROOT = pathlib.Path("outputs/bao2d_json"); OUTROOT.mkdir(parents=True, exist_ok=True)

# Small grid for verification; expand once working.
GAMMAS = np.round(np.linspace(1.54, 1.60, 4), 3)
FS      = np.round(np.linspace(2.50, 2.54, 5), 3)

BASE = {
  "mode": "bao-only",
  "plus_lya": True,
  "bao_csv": "data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv",
  "bao_cov": "data/desi_dr1_bao/bao_covariance_plus_lya.csv",
  "sn_csv":  "data/pantheon_plus/sn_MATCHED_mu.csv",
  "sn_cov":  "outputs/Pantheon+SH0ES_STAT+SYS.cal.cov",
  "free": ["A","phi"]
}

def _glob(patterns):
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(pat, recursive=True))
    return sorted(set(cands))

def _mt(p):
    try: return os.path.getmtime(p)
    except: return 0.0

def _extract_params(J):
    """
    Return a dict with numeric-ish 'gamma', 'f', possibly nested in J['best_params'].
    Also returns chi2 and (phi, A) if present.
    """
    def getnum(d, k):
        v = d.get(k)
        try: return float(v)
        except: return None

    bp = J.get("best_params") or J.get("params") or {}
    gamma = getnum(J, "gamma"); f = getnum(J, "f")
    if gamma is None: gamma = getnum(bp, "gamma")
    if f     is None: f     = getnum(bp, "f")

    chi2 = J.get("chi2") or J.get("chi_sq") or J.get("chisq") or J.get("chi2_min")
    try: chi2 = float(chi2)
    except: chi2 = None

    phi = getnum(bp, "phi")
    A   = getnum(bp, "A")
    return gamma, f, chi2, phi, A

def _recent_jsons(since_ts):
    # widen window a touch for slow FS
    return [p for p in _glob(["outputs/**/*.json"]) if _mt(p) >= since_ts - 1.0]

def pick_json_by_content(g, f, since_ts):
    """
    Search all recent JSONs, parse, and pick the one whose (gamma,f) match.
    Matching logic:
      - prefer |gamma-g|<=1e-3 and |f-f_in|<=1e-3
      - if multiple, pick newest
      - else fall back to newest recent JSON (with a warning)
    """
    cands = []
    recents = _recent_jsons(since_ts)
    for p in recents:
        try:
            with open(p) as fh: J = json.load(fh)
        except Exception:
            continue
        gamma, ff, chi2, phi, A = _extract_params(J)
        score = None
        if gamma is not None and ff is not None:
            if abs(gamma - g) <= 1e-3 and abs(ff - f) <= 1e-3:
                score = 2  # strong match
            elif abs(gamma - g) <= 1e-2 and abs(ff - f) <= 1e-2:
                score = 1  # weak match
        if score is not None:
            cands.append((score, _mt(p), p, chi2, phi, A))
    if cands:
        # prefer strong match (score high), then newest
        cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
        _, _, path, chi2, phi, A = cands[0]
        return path, chi2, phi, A

    # No content match — fall back to newest recent JSON (diagnostic warning)
    recents = [(p, _mt(p)) for p in recents]
    if not recents:
        return None, None, None, None
    recents.sort(key=lambda t: t[1], reverse=True)
    path = recents[0][0]
    try:
        J = json.load(open(path))
        _, _, chi2, phi, A = _extract_params(J)
    except Exception:
        chi2 = phi = A = None
    sys.stderr.write(
        f"\n[WARN] No JSON reported matching g={g:.3f}, f={f:.3f} by content.\n"
        f"       Falling back to newest recent JSON: {path}\n"
        f"       Consider grepping for 'out_prefix' in joint_guard.py.\n"
    )
    return path, chi2, phi, A

def run_one(g, f):
    prefix = OUTROOT / f"g{g:.3f}_f{f:.3f}"
    cfg = dict(BASE)
    cfg["gamma"] = float(g)
    cfg["f"]     = float(f)
    cfg["out_prefix"] = str(prefix)
    cfg["out"] = str(prefix) + ".json"  # harmless if runner ignores

    run_started_at = time.time()
    # Drive the runner
    sp.run(["python3","tools/run_with_cfg.py", json.dumps(cfg), str(prefix)+".json"], check=True)

    # Try for up to ~3s with backoff; then content-match across outputs/**
    for delay in (0.05, 0.10, 0.20, 0.40, 0.80, 1.20):
        # quick local/filename guesses first (cheap)
        base = prefix.name
        local = _glob([str(prefix) + "*.json", str(prefix) + ".json"])
        tree  = [] if local else _glob([f"outputs/**/*{base}*.json", f"outputs/**/{base}.json"])
        picks = local or tree
        if picks:
            picks.sort(key=_mt, reverse=True)
            p = picks[0]
            try:
                with open(p) as fh: J=json.load(fh)
                gamma, ff, chi2, phi, A = _extract_params(J)
                return {
                    "gamma": g, "f": f, "chi2": float(chi2),
                    "phi": phi, "A": A,
                    "json": os.path.relpath(p),
                }
            except Exception:
                pass
        time.sleep(delay)

    # Final fallback: search by JSON content across outputs/**
    p, chi2, phi, A = pick_json_by_content(g, f, run_started_at)
    if not p:
        # Print diagnostics: list 20 newest JSONs regardless
        now = time.time()
        recent = [(q, _mt(q)) for q in _glob(["outputs/**/*.json"])]
        recent.sort(key=lambda qt: qt[1], reverse=True)
        sys.stderr.write(
            f"\n[ERR] No JSON found for g={g:.3f}, f={f:.3f} even by content scan.\n"
            f"     Newest JSONs under outputs/:\n"
        )
        for q,t in recent[:20]:
            ts = datetime.fromtimestamp(t).strftime("%H:%M:%S")
            sys.stderr.write(f"       {ts}  {q}\n")
        raise FileNotFoundError(f"No JSON emitted (name- or content-match) for g={g:.3f}, f={f:.3f}.")

    if chi2 is None:
        raise RuntimeError(f"Picked {p} but couldn't read chi2 — open it and tell me the key names.")
    return {"gamma": g, "f": f, "chi2": float(chi2), "phi": phi, "A": A, "json": os.path.relpath(p)}

def main():
    rows=[]
    for g in GAMMAS:
        for f in FS:
            rec = run_one(g, f)
            print(f"[ok] g={g:.3f} f={f:.3f} -> {os.path.basename(rec['json'])}  chi2={rec['chi2']:.4f}")
            rows.append(rec)

    outcsv = OUTROOT / "summary.csv"
    with open(outcsv, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["gamma","f","chi2","phi","A","json"])
        w.writeheader(); w.writerows(rows)
    print("Wrote", outcsv)

if __name__ == "__main__":
    main()

