#!/usr/bin/env python3
"""
Emit <prefix>.metric.txt with "chi2: <value>" if we can infer a chi^2-like metric.
Also print a concise probe report so you can see what's nearby.

Usage:
  python3 tools/emit_metric_beacon_probe.py outputs/phase8/joint_kappa0
"""
from __future__ import annotations
import sys, os, json, csv, glob, math, re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional

STEMS = tuple("chi chisq chi_sq chi2 chi^2 χ2 χ² -2logL neg2logL minus2logL minus2loglike nll loglike lnL logL rss sse".split())
NESTED = ("fit","result","summary","stats","metrics","breakdown","report")
PREF_JSON = (".json", ".normalized.json")
CSV_PREF_COLS = ("chi2","chi2_i","delta_chi2","resid2","res2","nll","minus2logL","neg2logL")
CSV_FUZZ = re.compile(r"(chi|chisq|chi[_ ]?square|resid2|res2|nll|neg2log|minus2log|loglike|lnL)", re.I)

def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return None

def num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x,(int,float)) and math.isfinite(x) else None

def coerce_to_chi2(v: float, key: str) -> float:
    k = key.lower()
    if "nll" in k:              return 2.0*v
    if "logl" in k or "lnl" in k: return -2.0*v
    # rss/sse treated as chi^2-scale already
    return v

def dig_numeric_candidates(d: Dict[str, Any], prefix="") -> Dict[str,float]:
    out = {}
    for k,v in d.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(dig_numeric_candidates(v, kk))
        elif isinstance(v, (int,float)) and math.isfinite(v):
            out[kk] = float(v)
    return out

def try_json(stem: str) -> Tuple[Optional[float], str, str]:
    # return (chi2, source_path, note)
    for suf in PREF_JSON:
        p = f"{stem}{suf}"
        if not os.path.exists(p): continue
        J = load_json(p)
        if not isinstance(J, dict): continue

        # 1) Targeted: breakdown sums
        for sect in NESTED:
            sub = J.get(sect)
            if isinstance(sub, dict):
                s = 0.0; seen = False
                for k in ("chi2","chi2_total","best_chi2","chi2_min",
                          "chi2_bao","chi2_sn","chi2_lya","chi2_other",
                          "chisq","chi_sq","rss","sse"):
                    v = sub.get(k)
                    if isinstance(v,(int,float)):
                        s += float(v); seen = True
                if seen:
                    return s, p, f"{sect} sum"

                # NLL/loglike conversions inside the section
                for k in sub.keys():
                    if isinstance(sub[k], (int,float)):
                        if re.search(r"nll|neg.?log", k, re.I):
                            return 2.0*float(sub[k]), p, f"{sect}.{k}→2*NLL"
                        if re.search(r"loglike|lnL|logL", k, re.I):
                            return -2.0*float(sub[k]), p, f"{sect}.{k}→-2*lnL"

        # 2) Top-level direct / conversions
        for k in J.keys():
            v = J[k]
            y = num(v)
            if y is None: continue
            if re.search(r"^(chi2|chisq|chi_sq|chi2_total|best_chi2|chi2_min|rss|sse|-?2logL|neg2logL|minus2logL|minus2loglike)$", k, re.I):
                return coerce_to_chi2(y,k), p, f"top.{k}"

        # 3) Fuzzy: any numeric key containing chi / logL / nll etc.
        cand = dig_numeric_candidates(J)
        # strong preference: keys containing 'chi'
        chi_like = {k:v for k,v in cand.items() if re.search(r"chi", k, re.I)}
        if chi_like:
            # if multiple, prefer keys ending with chi2/chi2_total/best_chi2, else max magnitude
            pri = sorted(chi_like.items(), key=lambda kv: (not re.search(r"(best_?chi2|chi2(_total)?$)", kv[0], re.I), -abs(kv[1])))
            k,v = pri[0]
            return v, p, f"fuzzy:{k}"
        # otherwise look for NLL/loglike and convert
        for k,v in cand.items():
            if re.search(r"nll|neg.?log", k, re.I):
                return 2.0*v, p, f"fuzzy:{k}→2*NLL"
            if re.search(r"loglike|lnL|logL", k, re.I):
                return -2.0*v, p, f"fuzzy:{k}→-2*lnL"
    return None, "", ""

def try_csvs(stem: str) -> Tuple[Optional[float], str, str]:
    csvs = sorted(glob.glob(stem+"*.csv"))
    for p in csvs:
        try:
            with open(p, newline="") as f:
                rdr = csv.DictReader(f)
                fields = rdr.fieldnames or []
                # choose preferred column
                cols = [c for c in fields if c in CSV_PREF_COLS]
                if not cols:
                    cols = [c for c in fields if CSV_FUZZ.search(c or "")]
                if not cols: 
                    continue
                # compute sums for each candidate col
                sums = {c:0.0 for c in cols}
                counts = {c:0 for c in cols}
                for r in rdr:
                    for c in cols:
                        try:
                            sums[c] += float(r.get(c,""))
                            counts[c] += 1
                        except: pass
                # pick best column by exactness priority
                for pref in ("chi2","chi2_i","delta_chi2","resid2","res2","nll","minus2logL","neg2logL"):
                    if pref in sums and counts[pref] > 0:
                        val = sums[pref]
                        if pref == "nll": val *= 2.0
                        return val, p, f"csv:{pref} sum (n={counts[pref]})"
                # else take the first viable
                for c in cols:
                    if counts[c] > 0:
                        val = sums[c]
                        if re.search(r"nll", c, re.I): val *= 2.0
                        return val, p, f"csv:{c} sum (n={counts[c]})"
        except Exception:
            continue
    return None, "", ""

def main(stem: str) -> int:
    print(f"[probe] stem: {stem}")
    chi2, src, note = try_json(stem)
    if isinstance(chi2,(int,float)):
        out = f"{stem}.metric.txt"
        with open(out,"w") as f: f.write(f"chi2: {chi2}\n")
        print(f"[probe] WROTE {out}  ← {src}  ({note})   chi2={chi2}")
        return 0
    chi2, src, note = try_csvs(stem)
    if isinstance(chi2,(int,float)):
        out = f"{stem}.metric.txt"
        with open(out,"w") as f: f.write(f"chi2: {chi2}\n")
        print(f"[probe] WROTE {out}  ← {src}  ({note})   chi2={chi2}")
        return 0

    # nothing found — print neighbors + top keys to guide next step
    print("[probe] no χ²-like metric found. Neighbors:")
    for suf in PREF_JSON:
        p = f"{stem}{suf}"
        if os.path.exists(p):
            print("   json:", p)
            J = load_json(p)
            if isinstance(J, dict):
                keys = list(J.keys())
                print("     top keys:", keys[:20])
                for sect in NESTED:
                    if isinstance(J.get(sect), dict):
                        print(f"     {sect}.keys:", list(J[sect].keys())[:20])
    for p in sorted(glob.glob(stem+"*")):
        if p.endswith((".json",".csv",".txt",".log",".png",".pdf",".npy",".npz")):
            print("   file:", os.path.basename(p))
    print("[probe] If a likely metric key shows above (e.g. breakdown.total, metrics.cost), tell me its path and I’ll wire it in — still no repo patches required.")
    return 3

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tools/emit_metric_beacon_probe.py <out_prefix_stem>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
