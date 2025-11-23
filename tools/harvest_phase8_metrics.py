#!/usr/bin/env python3
"""
Phase-8 metric sweeper (v2): ALWAYS produces a table/CSV.

Priority per stem:
  1) --metric (CLI override) if provided for that stem
  2) <stem>.override          (text: 'metric: <num>' + optional 'kind: KIND')
  3) <stem>.fit.json          (chi2 or metric/kind)
  4) <stem>.normalized.json   (chi2 / breakdown sums / AIC/BIC / NLL→2*NLL / lnL→-2lnL)
  5) <stem>.metric*.txt       (metric/kind or chi2)
  6) JSON['log_file']         (scrape chi2/AIC/BIC/NLL/lnL lines)
  7) Neighbor CSVs            (sum chi2/chi2_i/resid2 or 2*nll)
  8) --metric-key path.to.key (force-extract a value from JSON if present)
  9) LAST RESORT: prompt-like fallback if --default-kind/--default-value provided

Outputs:
  - Console table
  - CSV: outputs/phase8/phase8_metrics.csv  (stem, metric, kind, source)
  - Deltas vs first stem

Usage examples:
  # Pure autodetect:
  python3 tools/harvest_phase8_metrics.py outputs/phase8/joint_baseline outputs/phase8/joint_tau0

  # If nothing is found, you can set explicit values (one or more):
  python3 tools/harvest_phase8_metrics.py \
      --metric outputs/phase8/joint_baseline=301.42 \
      --metric outputs/phase8/joint_tau0=307.11 \
      outputs/phase8/joint_baseline outputs/phase8/joint_tau0

  # If JSON has the value under a known key:
  python3 tools/harvest_phase8_metrics.py \
      --metric-key chi2 \
      outputs/phase8/joint_kappa0 outputs/phase8/joint_A0

  # If you want a default fallback (when nothing else is found):
  python3 tools/harvest_phase8_metrics.py \
      --default-value 0 --default-kind UNKNOWN \
      outputs/phase8/joint_baseline outputs/phase8/joint_tau0
"""
from __future__ import annotations
import sys, os, re, json, csv, glob, math, argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

KIND_ORDER = ["CHI2","M2LOG","NLL2","AIC","BIC","RSS","SSE","CSV_SUM","UNKNOWN"]

def is_num(x): 
    if isinstance(x,(int,float)): return math.isfinite(x)
    if isinstance(x,str):
        try:
            v=float(x.strip())
            return math.isfinite(v)
        except: return False
    return False

def to_num(x):
    if isinstance(x,(int,float)): return float(x)
    if isinstance(x,str):
        return float(x.strip())
    raise ValueError

def read_txt_kv(path):
    kv={}
    try:
        for ln in open(path, "r", errors="ignore"):
            if ":" in ln:
                k,v = ln.split(":",1)
                kv[k.strip().lower()] = v.strip()
    except Exception:
        pass
    return kv

def read_json(path) -> Optional[dict]:
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

def dig_nums(d, prefix=""):
    out={}
    if not isinstance(d,dict): return out
    for k,v in d.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v,dict): out.update(dig_nums(v,kk))
        elif is_num(v): out[kk]=to_num(v)
    return out

def coerce_pair(val, kind_guess):
    # Normalize kind name
    kind = (kind_guess or "UNKNOWN").upper()
    return val, kind

def from_cli_override(stem, cli_metrics: Dict[str, float], default_kind: str):
    if stem in cli_metrics:
        return cli_metrics[stem], default_kind.upper(), f"cli:{stem}"

def from_override_file(stem):
    p=f"{stem}.override"
    if not os.path.exists(p): return None
    kv=read_txt_kv(p)
    if "metric" in kv and is_num(kv["metric"]):
        kind=(kv.get("kind") or "UNKNOWN").upper()
        return to_num(kv["metric"]), kind, f"override:{os.path.basename(p)}"
    return None

def from_fit_json(stem):
    p = f"{stem}.fit.json"
    J = read_json(p)
    if not isinstance(J,dict): return None
    if is_num(J.get("chi2")):
        return to_num(J["chi2"]), "CHI2", f"fit:{os.path.basename(p)}"
    if is_num(J.get("metric")):
        kind = (J.get("kind") or "UNKNOWN").upper()
        return to_num(J["metric"]), kind, f"fit:{os.path.basename(p)}"
    return None

def from_normalized_json(stem, metric_key=None):
    # returns tuple or None; also pass back log path to use if needed
    for suf in (".normalized.json",".json"):
        p=f"{stem}{suf}"
        J=read_json(p)
        if not isinstance(J,dict): continue
        # explicit metric key if provided (supports nested path a.b.c)
        if metric_key:
            cur=J
            try:
                for part in metric_key.split("."):
                    cur = cur[part]
                if is_num(cur):
                    return to_num(cur), "UNKNOWN", f"json:{os.path.basename(p)}::{metric_key}", J.get("log_file","")
            except Exception:
                pass

        for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min"]:
            if is_num(J.get(k)):
                return to_num(J[k]), "CHI2", f"json:{os.path.basename(p)}::{k}", J.get("log_file","")

        for sect in ["fit","result","summary","stats","metrics","breakdown","report"]:
            sub=J.get(sect)
            if isinstance(sub,dict):
                for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min"]:
                    v=sub.get(k)
                    if is_num(v): return to_num(v), "CHI2", f"json:{os.path.basename(p)}::{sect}.{k}", J.get("log_file","")
                if sect=="breakdown":
                    s=0.0; seen=False
                    for k in ["chi2_bao","chi2_sn","chi2_lya","chi2_other"]:
                        if is_num(sub.get(k)): s+=to_num(sub[k]); seen=True
                    if seen: return s, "CHI2", f"json:{os.path.basename(p)}::{sect}.sum", J.get("log_file","")
                # conversions
                for k in list(sub.keys()):
                    v=sub.get(k)
                    if is_num(v) and re.search(r"nll|neg.?log", k, re.I): 
                        return 2.0*to_num(v), "NLL2", f"json:{os.path.basename(p)}::{sect}.{k}", J.get("log_file","")
                    if is_num(v) and re.search(r"loglike|lnL|logL", k, re.I):
                        return -2.0*to_num(v), "M2LOG", f"json:{os.path.basename(p)}::{sect}.{k}", J.get("log_file","")
        for k,kind in [("AIC","AIC"),("BIC","BIC"),("rss","RSS"),("sse","SSE")]:
            v=J.get(k)
            if is_num(v): return to_num(v), kind, f"json:{os.path.basename(p)}::{k}", J.get("log_file","")
        # fuzzy chi anywhere
        for k,v in dig_nums(J).items():
            if re.search(r"chi", k, re.I):
                return v, "CHI2", f"json:{os.path.basename(p)}::fuzzy:{k}", J.get("log_file","")
        return None  # tried this JSON; move to the next
    return None

def from_metric_txt(stem):
    for p in sorted(glob.glob(stem+".metric*.txt")):
        kv=read_txt_kv(p)
        if "chi2" in kv and is_num(kv["chi2"]):
            return to_num(kv["chi2"]), "CHI2", f"txt:{os.path.basename(p)}"
        if "metric" in kv and is_num(kv["metric"]):
            kind=(kv.get("kind") or "UNKNOWN").upper()
            return to_num(kv["metric"]), kind, f"txt:{os.path.basename(p)}"
    return None

def from_log_path(log_path):
    if not (isinstance(log_path,str) and os.path.exists(log_path)): return None
    txt=open(log_path,"r",errors="ignore").read()
    lines=[ln for ln in txt.splitlines() if re.search(r"(RESULT|AIC|BIC|chi|nll|loglike|lnL)", ln, re.I)]
    RX = [
        (re.compile(r"(?:^|\s)(?:chi\s*\^?\s*2|chi2|chisq|χ\s*2|χ²|chi[_ -]?square)\s*[:=]\s*([0-9eE+\-\.]+)", re.I), "CHI2", lambda v: float(v)),
        (re.compile(r"(?:^|\s)(?:-?\s*2\s*(?:ln\s*L|logL)|neg2logL|minus2logL|minus2loglike)\s*[:=]\s*([0-9eE+\-\.]+)", re.I), "M2LOG", lambda v: float(v)),
        (re.compile(r"(?:^|\s)(?:nll|neg(?:ative)?\s*log(?:like|likelihood))\s*[:=]\s*([0-9eE+\-\.]+)", re.I), "NLL2", lambda v: 2.0*float(v)),
        (re.compile(r"Δ?\s*AIC\s*=\s*([0-9eE+\-\.]+)", re.I), "AIC",  lambda v: float(v)),
        (re.compile(r"Δ?\s*BIC\s*=\s*([0-9eE+\-\.]+)", re.I), "BIC",  lambda v: float(v)),
    ]
    for rx,kind,conv in RX:
        for ln in lines:
            m=rx.search(ln)
            if m:
                try:
                    val=conv(m.group(1))
                    if math.isfinite(val): return val, kind, f"log:{os.path.basename(log_path)}"
                except: pass
    for rx,kind,conv in RX:
        m=rx.search(txt)
        if m:
            try:
                val=conv(m.group(1))
                if math.isfinite(val): return val, kind, f"log:{os.path.basename(log_path)}"
            except: pass
    return None

def from_csvs(stem):
    for p in sorted(glob.glob(stem+"*.csv")):
        try:
            with open(p, newline="") as f:
                rdr=csv.DictReader(f)
                fields=rdr.fieldnames or []
                order = [c for c in ("chi2","chi2_i","delta_chi2","resid2","res2","nll","minus2logL","neg2logL") if c in fields]
                if not order:
                    order=[c for c in fields if re.search(r"(chi|resid2|res2|nll|neg2log|minus2log|loglike|lnL)", c, re.I)]
                if not order: continue
                sums={c:0.0 for c in order}; seen={c:0 for c in order}
                for r in rdr:
                    for c in order:
                        try: sums[c]+=float(r.get(c,"")); seen[c]+=1
                        except: pass
                for c in order:
                    if seen[c]:
                        val=sums[c]; kind="CSV_SUM"
                        if re.fullmatch(r"nll", c, re.I): val,kind=2.0*val,"NLL2"
                        if re.search(r"(minus2logL|neg2logL)", c, re.I): kind="M2LOG"
                        return val, kind, f"csv:{os.path.basename(p)}:{c}"
        except Exception:
            pass
    return None

def harvest_one(stem, cli_metrics, metric_key, default_value, default_kind):
    # 1) CLI override
    r = from_cli_override(stem, cli_metrics, default_kind or "UNKNOWN")
    if r: return stem, r[0], r[1], r[2]
    # 2) .override
    r = from_override_file(stem)
    if r: return stem, r[0], r[1], r[2]
    # 3) fit.json
    r = from_fit_json(stem)
    if r: return stem, r[0], r[1], r[2]
    # 4) normalized/json
    r = from_normalized_json(stem, metric_key=metric_key)
    if r:
        v,k,src,*rest = r
        # 5) metric*.txt (prefer if exists and explicit)
        rt = from_metric_txt(stem)
        if rt: return stem, rt[0], rt[1], rt[2]
        # else accept normalized/json result
        return stem, v, k, src
    # 5) metric*.txt (standalone)
    r = from_metric_txt(stem)
    if r: return stem, r[0], r[1], r[2]
    # 6) try log (if we can discover it from .normalized.json)
    for suf in (".normalized.json",".json",".fit.json"):
        p=f"{stem}{suf}"
        J=read_json(p)
        if isinstance(J,dict):
            log=J.get("log_file")
            r = from_log_path(log)
            if r: return stem, r[0], r[1], r[2]
    # 7) csv neighbors
    r = from_csvs(stem)
    if r: return stem, r[0], r[1], r[2]
    # 8) default fallback
    if default_value is not None:
        return stem, default_value, (default_kind or "UNKNOWN").upper(), "default"
    # 9) give up
    return stem, None, "UNKNOWN", ""

def parse_cli_metrics(pairs):
    out={}
    for s in pairs:
        if "=" not in s:
            print(f"[warn] --metric expects STEM=VALUE (got {s})")
            continue
        k,v = s.split("=",1)
        try:
            out[k]=float(v)
        except:
            print(f"[warn] non-numeric metric for {k}: {v}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="Phase-8 stems like outputs/phase8/joint_baseline")
    ap.add_argument("--metric", action="append", default=[], help="Override metric: STEM=VALUE")
    ap.add_argument("--metric-key", default=None, help="JSON key path to use if present (e.g., 'breakdown.chi2')")
    ap.add_argument("--default-value", type=float, default=None, help="Fallback metric if nothing is found")
    ap.add_argument("--default-kind", default="UNKNOWN", help="Kind label for overrides/defaults (default: UNKNOWN)")
    args=ap.parse_args()

    cli_metrics = parse_cli_metrics(args.metric)
    rows=[]
    for s in args.stems:
        stem, v, kind, src = harvest_one(s, cli_metrics, args.metric_key, args.default_value, args.default_kind)
        rows.append((stem, v, kind, src))

    out_csv = "outputs/phase8/phase8_metrics.csv"
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["stem","metric","kind","source"])
        for stem,v,kind,src in rows:
            w.writerow([stem, "" if v is None else f"{v:.12g}", kind, src])

    wstem = max(len(os.path.basename(s)) for s in args.stems)
    print("\n[Phase-8 metrics]")
    print(f"{'stem'.ljust(wstem)}  {'metric':>14}  {'kind':<8}  source")
    print("-"*(wstem+2+14+2+8+2+20))
    for stem,v,kind,src in rows:
        name=os.path.basename(stem); m=f"{v:.6g}" if v is not None else "—"
        print(f"{name.ljust(wstem)}  {m:>14}  {kind:<8}  {src}")
    print(f"\nWrote {out_csv}")

    base = rows[0][1]
    if base is not None:
        print("\n[Deltas vs first]")
        for i,(stem,v,kind,src) in enumerate(rows):
            name=os.path.basename(stem)
            if i==0 or v is None:
                print(f"{name}: —")
            else:
                print(f"{name}: Δ={v-base:+.6g}   (kind={kind})")

if __name__=="__main__":
    main()
