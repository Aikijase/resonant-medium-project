#!/usr/bin/env python3
"""
Emit <stem>.metric.txt with a single line:   metric: <value>   kind: <KIND>
…where KIND is one of: CHI2, AIC, BIC, NLL2 (2*NLL), M2LOG (−2lnL), RSS, SSE, CSV_SUM, UNKNOWN

Priority:
  (a) JSON neighbors: <stem>.json / <stem>.normalized.json
  (b) LOG pointed to by JSON['log_file']
  (c) CSV neighbors: sum of chi2/chi2_i/resid2/nll/etc.

Usage:
  python3 tools/emit_metric_beacon_any.py outputs/phase8/joint_kappa0
"""
from __future__ import annotations
import sys, os, re, json, csv, glob, math

PREF_JSON = (".json", ".normalized.json")
NESTED = ("fit","result","summary","stats","metrics","breakdown","report")

# ---------- helpers ----------
def load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return None

def is_num(x):
    return isinstance(x,(int,float)) and math.isfinite(x)

def dig_nums(d, prefix=""):
    out={}
    for k,v in (d.items() if isinstance(d,dict) else []):
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): out.update(dig_nums(v, kk))
        elif is_num(v): out[kk]=float(v)
    return out

def pick_json_metric(J):
    # direct chi2-ish first
    for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min"]:
        v=J.get(k)
        if is_num(v): return float(v),"CHI2",f"json:{k}"
    # nested blocks
    for sect in NESTED:
        sub=J.get(sect)
        if isinstance(sub,dict):
            for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min"]:
                v=sub.get(k)
                if is_num(v): return float(v),"CHI2",f"json:{sect}.{k}"
            # sums of breakdown (bao/sn/lya/other)
            s=0.0; seen=False
            for k in ["chi2_bao","chi2_sn","chi2_lya","chi2_other"]:
                if is_num(sub.get(k)): s+=float(sub[k]); seen=True
            if seen: return s,"CHI2",f"json:{sect}.sum"
            # AIC/BIC if present inside
            for k,kind in [("AIC","AIC"),("BIC","BIC")]:
                v=sub.get(k); 
                if is_num(v): return float(v),kind,f"json:{sect}.{k}"
            # NLL/loglike conversions
            for k in sub:
                if is_num(sub[k]) and re.search(r"nll|neg.?log", k, re.I): return 2.0*float(sub[k]),"NLL2",f"json:{sect}.{k}"
                if is_num(sub[k]) and re.search(r"loglike|lnL|logL", k, re.I): return -2.0*float(sub[k]),"M2LOG",f"json:{sect}.{k}"
    # top-level AIC/BIC
    for k,kind in [("AIC","AIC"),("BIC","BIC")]:
        v=J.get(k); 
        if is_num(v): return float(v),kind,f"json:{k}"
    # rss/sse
    for k,kind in [("rss","RSS"),("sse","SSE")]:
        v=J.get(k); 
        if is_num(v): return float(v),kind,f"json:{k}"
    # fuzzy: any key containing "chi"
    for k,v in dig_nums(J).items():
        if re.search(r"chi", k, re.I): return v,"CHI2",f"json:fuzzy:{k}"
    # fuzzy: NLL/loglike
    for k,v in dig_nums(J).items():
        if re.search(r"nll|neg.?log", k, re.I): return 2.0*v,"NLL2",f"json:fuzzy:{k}"
        if re.search(r"loglike|lnL|logL", k, re.I): return -2.0*v,"M2LOG",f"json:fuzzy:{k}"
    return None,None,None

RX_LOG = [
    # chi2-ish
    (re.compile(r"(?:^|\s)(?:chi\s*\^?\s*2|chi2|chisq|χ\s*2|χ²|chi[_ -]?square)\s*[:=]\s*([0-9eE+\-\.]+)", re.I), ("CHI2", lambda v: float(v))),
    # -2 ln L family
    (re.compile(r"(?:^|\s)(?:-?\s*2\s*(?:ln\s*L|logL)|neg2logL|minus2logL|minus2loglike)\s*[:=]\s*([0-9eE+\-\.]+)", re.I), ("M2LOG", lambda v: float(v))),
    # NLL → 2*NLL
    (re.compile(r"(?:^|\s)(?:nll|neg(?:ative)?\s*log(?:like|likelihood))\s*[:=]\s*([0-9eE+\-\.]+)", re.I), ("NLL2", lambda v: 2.0*float(v))),
    # AIC / BIC lines (accept ΔAIC or AIC value)
    (re.compile(r"Δ?\s*AIC\s*=\s*([0-9eE+\-\.]+)", re.I), ("AIC", lambda v: float(v))),
    (re.compile(r"Δ?\s*BIC\s*=\s*([0-9eE+\-\.]+)", re.I), ("BIC", lambda v: float(v))),
]

def pick_log_metric(path):
    try:
        txt=open(path,"r",errors="ignore").read()
    except Exception:
        return None,None,None
    # try to find RESULT-style line first to keep it consistent
    lines = [ln for ln in txt.splitlines() if re.search(r"RESULT|AIC|BIC|chi|nll|loglike|lnL", ln, re.I)]
    for rx,(kind,conv) in RX_LOG:
        for ln in lines:
            m=rx.search(ln)
            if m:
                try:
                    val=conv(m.group(1))
                    if math.isfinite(val): return val,kind,f"log:{os.path.basename(path)}"
                except Exception:
                    pass
    # last resort: full text scan
    for rx,(kind,conv) in RX_LOG:
        m=rx.search(txt)
        if m:
            try:
                val=conv(m.group(1))
                if math.isfinite(val): return val,kind,f"log:{os.path.basename(path)}"
            except Exception:
                pass
    return None,None,None

def pick_csv_metric(stem):
    for p in sorted(glob.glob(stem+"*.csv")):
        try:
            with open(p, newline="") as f:
                rdr=csv.DictReader(f)
                fields=rdr.fieldnames or []
                # prefer exact names first
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
                        val=sums[c]
                        kind="CSV_SUM"
                        if re.search(r"^nll$", c, re.I): val,kind=2.0*val,"NLL2"
                        if re.search(r"minus2logL|neg2logL", c, re.I): kind="M2LOG"
                        return val,kind,f"csv:{os.path.basename(p)}:{c}"
        except Exception:
            pass
    return None,None,None

def main(stem):
    # 1) JSON
    J=None; jpath=None
    for suf in PREF_JSON:
        p=f"{stem}{suf}"
        if os.path.exists(p):
            J=load_json(p); jpath=p; break
    if J:
        val,kind,src=pick_json_metric(J)
        if is_num(val):
            with open(f"{stem}.metric.txt","w") as f:
                f.write(f"metric: {val}\nkind: {kind}\n")
            print("WROTE", f"{stem}.metric.txt", "←", src, f"({kind})", "val=", val)
            return 0
        # try its log
        log=J.get("log_file")
        if isinstance(log,str) and os.path.exists(log):
            val,kind,src=pick_log_metric(log)
            if is_num(val):
                with open(f"{stem}.metric.txt","w") as f:
                    f.write(f"metric: {val}\nkind: {kind}\n")
                print("WROTE", f"{stem}.metric.txt", "←", src, f"({kind})", "val=", val)
                return 0
    # 2) CSV neighbors
    val,kind,src=pick_csv_metric(stem)
    if is_num(val):
        with open(f"{stem}.metric.txt","w") as f:
            f.write(f"metric: {val}\nkind: {kind}\n")
        print("WROTE", f"{stem}.metric.txt", "←", src, f"({kind})", "val=", val)
        return 0

    print(f"[emit_metric_beacon_any] no metric found near {stem}", file=sys.stderr)
    return 3

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python3 tools/emit_metric_beacon_any.py <out_prefix_stem>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
