#!/usr/bin/env python3
"""
Emit <prefix>.metric.txt with a line:  chi2: <value>

Where it looks (in order):
  1) <prefix>.json, then <prefix>.normalized.json
     - Direct: chi2 / chisq / chi_sq / chi2_total / best_chi2 / chi2_min
     - Nested: ... inside {fit,result,summary,stats,metrics,breakdown}
     - Sum:    breakdown.chi2_bao + breakdown.chi2_sn (+ breakdown.chi2_lya if present)
     - Convert: nll -> 2*nll ; loglike/lnL -> -2*lnL ; rss/sse -> as-is (χ²-scale)
  2) Neighbor CSVs: sum of per-point columns 'chi2', 'chi2_i', or 'resid2',
     or 2*sum('nll') if only NLL given.

Usage:
  python3 tools/emit_metric_beacon_v2.py outputs/phase8/joint_kappa0
"""
import sys, os, json, csv, glob
from pathlib import Path
from typing import Any, Dict, Iterable

CAND_KEYS = (
    "chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min",
    "-2logL","neg2logL","minus2logL","minus2loglike","rss","sse"
)
NLL_KEYS = ("nll","negloglike","neg_loglike","neglogL","neg_logL")
LOGL_KEYS = ("loglike","log_likelihood","lnL","logL")
NESTED = ("fit","result","summary","stats","metrics","breakdown")

def load_json(path: str):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return None

def first_num(d: Dict[str, Any], keys: Iterable[str]):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)): return float(v)
    return None

def coerce_chi2_from_dict(d: Dict[str, Any]):
    # direct at top-level
    v = first_num(d, CAND_KEYS)
    if v is not None: return v
    # nested sections
    for sect in NESTED:
        sub = d.get(sect)
        if isinstance(sub, dict):
            v = first_num(sub, CAND_KEYS)
            if v is not None: return v
            # conversions in nested section
            nv = first_num(sub, NLL_KEYS)
            if nv is not None: return 2.0 * nv
            lv = first_num(sub, LOGL_KEYS)
            if lv is not None: return -2.0 * lv
            # common breakdown sum
            if sect == "breakdown":
                s = 0.0; seen = False
                for k in ("chi2_bao","chi2_sn","chi2_lya","chi2_other"):
                    if isinstance(sub.get(k), (int,float)):
                        s += float(sub[k]); seen = True
                if seen: return s
    # conversions at top-level
    nv = first_num(d, NLL_KEYS)
    if nv is not None: return 2.0 * nv
    lv = first_num(d, LOGL_KEYS)
    if lv is not None: return -2.0 * lv
    return None

def sum_csv_cols(path: str, cols: Iterable[str]):
    try:
        rows = list(csv.DictReader(open(path, newline="")))
    except Exception:
        return None
    if not rows: return None
    # single-row metrics?
    if len(rows) == 1:
        r = rows[0]
        for k in cols:
            if k in r:
                try: return float(r[k])
                except: pass
    # per-point contributions
    best = None
    for k in cols:
        s = 0.0; seen = False
        for r in rows:
            if k in r:
                try:
                    s += float(r[k]); seen = True
                except: pass
        if seen:
            # prefer explicit chi2 over resid2; keep the first viable
            if k in ("chi2","chi2_i"): return s
            best = s
    return best

def try_neighbor_csvs(stem: str):
    # look beside the stem for CSVs that might carry contributions
    # e.g., <stem>*.csv
    for p in sorted(glob.glob(stem + "*.csv")):
        # order of preference in csv
        val = sum_csv_cols(p, ("chi2","chi2_i","resid2","nll"))
        if val is not None:
            # nll sum needs *2
            if "nll" in (csv.DictReader(open(p)).fieldnames or []):
                val = 2.0 * val
            return val, p
    return None, None

def main(stem: str):
    # 1) JSON neighbors
    json_candidates = [f"{stem}.json", f"{stem}.normalized.json"]
    J = None; used = None
    for p in json_candidates:
        if os.path.exists(p):
            J = load_json(p)
            if J is not None:
                used = p
                chi2 = coerce_chi2_from_dict(J)
                if isinstance(chi2, (int,float)):
                    out = f"{stem}.metric.txt"
                    with open(out, "w") as f:
                        f.write(f"chi2: {chi2}\n")
                    print("WROTE", out, "←", used, "→ chi2:", chi2)
                    return 0
    # 2) CSV neighbors
    chi2, csvp = try_neighbor_csvs(stem)
    if isinstance(chi2, (int,float)):
        out = f"{stem}.metric.txt"
        with open(out, "w") as f:
            f.write(f"chi2: {chi2}\n")
        print("WROTE", out, "←", csvp, "→ chi2:", chi2)
        return 0
    print(f"[emit_metric_beacon_v2] no chi2-like metric found near {stem}", file=sys.stderr)
    return 3

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tools/emit_metric_beacon_v2.py <out_prefix_stem>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
