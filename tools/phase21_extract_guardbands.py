#!/usr/bin/env python3
"""
Phase-21 → canonical guardbands extractor (v2 heuristic).

Scans JSON files under --in-path (default: outputs/phase21), hunts for any combination
of omega^2-like arrays and matching lower/upper Kphi arrays (including left/right, min/max),
and emits a flat, per-point guardbands file:

  outputs/phase21/p21_guardbands_extracted.json  # list of {target, omega2, Kphi_lo, Kphi_hi}
  outputs/phase21/p21_guardbands_extracted.csv

Heuristics:
- Recognize omega keys: contains any of ["omega2","omega_sq","w2","omega^2","omega"].
- Recognize lo/hi keys: contains tokens like ["kphi","k","left","low","min"] and ["kphi","k","right","high","max"].
- Prefer pairs (lo/hi) with same length as omega array.
- Also accepts per-point dicts with any aliasing of those fields.

If multiple targets exist, tries to read a nearby "target" or "si_target" number;
otherwise defaults to 0.95.

Usage:
  python3 tools/phase21_extract_guardbands.py --in-path outputs/phase21
  # (Optional) target a specific file:
  python3 tools/phase21_extract_guardbands.py --in-json outputs/phase21/p21_guardbands.json
"""

import argparse, json, csv, pathlib, sys, re
from typing import Any, Dict, List, Tuple, Optional

Num = (int, float)

def is_num(x): return isinstance(x, (int, float)) and not isinstance(x, bool)
def looks_num_list(a): return isinstance(a, list) and len(a) > 0 and all(is_num(v) for v in a)

OMEGA_TOKENS = ["omega2","omega_sq","omega^2","w2","omega"]
K_TOKENS     = ["kphi","k"]
LO_TOKENS    = ["lo","low","left","min","lower"]
HI_TOKENS    = ["hi","high","right","max","upper"]

def key_contains_any(key: str, tokens: List[str]) -> bool:
    lk = key.lower()
    return any(t in lk for t in tokens)

def score_omega_key(k: str) -> int:
    s = 0
    lk = k.lower()
    for t in OMEGA_TOKENS: 
        if t in lk: s += 2
    if "2" in lk: s += 1
    return s

def score_k_key(k: str) -> int:
    s = 0
    lk = k.lower()
    if any(t in lk for t in K_TOKENS): s += 2
    if any(t in lk for t in LO_TOKENS + HI_TOKENS): s += 1
    return s

def find_numeric_arrays(obj: Any, path="") -> List[Tuple[str, List[float]]]:
    """Return list of (json_path, numeric_list)."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if looks_num_list(v):
                found.append((p, [float(x) for x in v]))
            else:
                found.extend(find_numeric_arrays(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            if looks_num_list(v):
                found.append((p, [float(x) for x in v]))
            else:
                found.extend(find_numeric_arrays(v, p))
    return found

def find_candidate_points(obj: Any, path="") -> List[Dict[str, Any]]:
    """Collect per-point dicts that might carry omega2 / Kphi_lo / Kphi_hi with aliases."""
    out = []
    if isinstance(obj, dict):
        # heuristic: small dict with numeric fields
        keys = list(obj.keys())
        if keys:
            kmap = {k.lower(): k for k in keys}
            # potential single-point mapping
            # pick omega candidate
            omega_key = None
            for k in keys:
                if is_num(obj[k]) and score_omega_key(k) > 0:
                    omega_key = k; break
            # pick lo/hi
            lo_key = None; hi_key = None
            for k in keys:
                lk = k.lower()
                if is_num(obj[k]) and (any(t in lk for t in LO_TOKENS) and any(t in lk for t in K_TOKENS)):
                    lo_key = k
                if is_num(obj[k]) and (any(t in lk for t in HI_TOKENS) and any(t in lk for t in K_TOKENS)):
                    hi_key = k
            if omega_key and lo_key and hi_key:
                tgt = obj.get("target", obj.get("si_target", obj.get("sync_index", 0.95)))
                out.append({
                    "target": float(tgt), "omega2": float(obj[omega_key]),
                    "Kphi_lo": float(obj[lo_key]), "Kphi_hi": float(obj[hi_key])
                })
        # recurse
        for v in obj.values(): out += find_candidate_points(v)
    elif isinstance(obj, list):
        for it in obj: out += find_candidate_points(it)
    return out

def group_by_length(arrs: List[Tuple[str, List[float]]]):
    d = {}
    for p, a in arrs:
        d.setdefault(len(a), []).append((p, a))
    return d

def guess_pairs_from_arrays(arrs: List[Tuple[str, List[float]]]) -> List[Dict[str, Any]]:
    """Given a pile of numeric arrays found anywhere, guess omega + (lo,hi) triplets of same length."""
    out = []
    if not arrs: return out
    bylen = group_by_length(arrs)
    # For each length bucket, try to find one omega and two K arrays
    for L, items in bylen.items():
        if L < 2: 
            continue
        # score omegas
        omegas = [(p,a,score_omega_key(p)) for p,a in items if score_omega_key(p)>0]
        # score K arrays into lo/hi candidates
        Ks = [(p,a,score_k_key(p)) for p,a in items if score_k_key(p)>0]
        if not omegas or len(Ks)<2: 
            continue
        # sort by score desc
        omegas.sort(key=lambda x: x[2], reverse=True)
        # build all distinct pairs of Ks; choose ones that look like lo/hi by name
        def is_lo_name(p): 
            lp=p.lower(); 
            return any(t in lp for t in LO_TOKENS) or "left" in lp or "min" in lp
        def is_hi_name(p): 
            lp=p.lower(); 
            return any(t in lp for t in HI_TOKENS) or "right" in lp or "max" in lp
        lo_candidates = [(p,a,s) for p,a,s in Ks if is_lo_name(p)]
        hi_candidates = [(p,a,s) for p,a,s in Ks if is_hi_name(p)]
        pairs = []
        if lo_candidates and hi_candidates:
            for lo in lo_candidates:
                for hi in hi_candidates:
                    if lo[0] != hi[0]:
                        pairs.append((lo,hi))
        else:
            # fallback: any two highest-scoring K arrays
            Ks_sorted = sorted(Ks, key=lambda x: x[2], reverse=True)
            for i in range(min(4, len(Ks_sorted))):
                for j in range(i+1, min(6, len(Ks_sorted))):
                    pairs.append((Ks_sorted[i], Ks_sorted[j]))
        # pick top omega and top pair
        omega = omegas[0]
        for (lo,hi) in pairs[:4]:  # try a few
            p_o, a_o, _ = omega
            p_l, a_l, _ = lo
            p_h, a_h, _ = hi
            if len(a_o)==len(a_l)==len(a_h):
                # fabricate rows with a default or inferred target (try to read a nearby scalar)
                tgt = 0.95
                out.append({"target": tgt, "omega2": a_o, "Kphi_lo": a_l, "Kphi_hi": a_h})
                break
    return out

def to_point_rows(entries: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    rows=[]
    for e in entries:
        if isinstance(e.get("omega2"), list):
            tgt = float(e.get("target", 0.95))
            for w2, lo, hi in zip(e["omega2"], e["Kphi_lo"], e["Kphi_hi"]):
                rows.append({"target": tgt, "omega2": float(w2), "Kphi_lo": float(lo), "Kphi_hi": float(hi)})
        else:
            rows.append({
                "target": float(e["target"]),
                "omega2": float(e["omega2"]),
                "Kphi_lo": float(e["Kphi_lo"]),
                "Kphi_hi": float(e["Kphi_hi"]),
            })
    return rows

def scan_json(path: pathlib.Path) -> List[Dict[str, float]]:
    try:
        data = json.load(path.open())
    except Exception:
        return []
    # First, see if per-point dicts are present anywhere
    pointish = find_candidate_points(data)
    if pointish:
        return to_point_rows(pointish)
    # Otherwise, scan numeric arrays and try to guess triplets
    arrs = find_numeric_arrays(data)
    guessed = guess_pairs_from_arrays(arrs)
    if guessed:
        return to_point_rows(guessed)
    return []

def auto_discover(dirpath: pathlib.Path) -> List[pathlib.Path]:
    files = []
    files += sorted(dirpath.glob("p21*guardband*.json"))
    files += sorted(dirpath.glob("**/p21*guardband*.json"))
    files += sorted(dirpath.glob("*.json"))
    files += sorted(dirpath.glob("**/*.json"))
    # de-dup while preserving order
    seen=set(); out=[]
    for p in files:
        if p.resolve() not in seen:
            out.append(p); seen.add(p.resolve())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", default="outputs/phase21")
    ap.add_argument("--in-json")
    args = ap.parse_args()

    base = pathlib.Path(args.in_path)
    targets = []
    if args.in_json:
        p = pathlib.Path(args.in_json)
        if p.exists():
            targets = [p]
    if not targets:
        targets = auto_discover(base)

    all_rows = []
    tried = 0
    for p in targets:
        if p.suffix.lower() != ".json":
            continue
        tried += 1
        rows = scan_json(p)
        if rows:
            all_rows.extend(rows)

    # Dedup and sort
    key = lambda r: (r["target"], r["omega2"], r["Kphi_lo"], r["Kphi_hi"])
    uniq = list({key(r): r for r in all_rows}.values())
    uniq.sort(key=lambda r: (r["target"], r["omega2"]))

    if not uniq:
        print(json.dumps({
            "status":"error",
            "message":"Heuristic scan found no usable omega2/Kphi_lo/Kphi_hi triplets.",
            "hint":"If Phase-21 only produced range summaries (omega2_min/max), re-run Phase-21 with per-point guardbands output enabled, or tell me the exact JSON shape."
        }, indent=2))
        sys.exit(2)

    outdir = base
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "p21_guardbands_extracted.json"
    out_csv  = outdir / "p21_guardbands_extracted.csv"

    with out_json.open("w") as f:
        json.dump(uniq, f, indent=2)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target","omega2","Kphi_lo","Kphi_hi"])
        w.writeheader()
        w.writerows(uniq)

    print(json.dumps({
        "status":"ok",
        "n_rows": len(uniq),
        "out_json": str(out_json),
        "out_csv": str(out_csv)
    }, indent=2))

if __name__ == "__main__":
    main()
