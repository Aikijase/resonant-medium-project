#!/usr/bin/env python3
import sys, json, glob, os, re, csv
from pathlib import Path

prefix = sys.argv[1] if len(sys.argv)>1 else "outputs/phase8/joint_baseline"

def list_candidates(pfx):
    bad = re.compile(r"(normalized|zero_pad_sweep|sweep_dt\d|mt_|surrogate_|loo_summary|cov_study|mock_delta_chi2|scan_penalty)")
    files = [f for f in glob.glob(pfx+"*") if not bad.search(os.path.basename(f))]
    return sorted(files, key=os.path.getmtime)

def try_from_json(path):
    try:
        J=json.load(open(path))
    except Exception:
        return None
    # direct chi2
    for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min"]:
        v=J.get(k); 
        if isinstance(v,(int,float)): return float(v)
    # nested
    for sect in ["fit","result","summary","stats","metrics"]:
        d=J.get(sect)
        if isinstance(d,dict):
            for kk in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min","neg2logL","minus2logL","-2logL","nll","loglike","lnL"]:
                v=d.get(kk)
                if isinstance(v,(int,float)):
                    if kk.lower() in ("nll",): return 2.0*float(v)
                    if "logl" in kk.lower() and kk.startswith("-2"): return float(v)
                    if kk in ("loglike","lnL"): return -2.0*float(v)
                    return float(v)
    # flat likelihood keys
    for kk in ["neg2loglike","minus2loglike","minus2logL","-2logL"]:
        v=J.get(kk)
        if isinstance(v,(int,float)): return float(v)
    for kk in ["nll","negloglike","neg_loglike","neglogL","neg_logL","minusloglike","minus_loglike"]:
        v=J.get(kk)
        if isinstance(v,(int,float)): return 2.0*float(v)
    for kk in ["loglike","log_likelihood","lnL","logL"]:
        v=J.get(kk)
        if isinstance(v,(int,float)): return -2.0*float(v)
    return None

def try_from_csv(path):
    try:
        with open(path, newline="") as f:
            R=list(csv.DictReader(f))
    except Exception:
        return None
    # common patterns: a single-row metrics CSV, or per-point nll/residuals
    if not R: return None
    # 1) single-row, chi2 column
    if len(R)==1:
        row=R[0]
        for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2"]:
            if k in row:
                try: return float(row[k])
                except: pass
        for k in ["neg2logL","-2logL","minus2logL"]:
            if k in row:
                try: return float(row[k])
                except: pass
        for k in ["nll","negloglike","neg_loglike"]:
            if k in row:
                try: return 2.0*float(row[k])
                except: pass
        for k in ["loglike","lnL"]:
            if k in row:
                try: return -2.0*float(row[k])
                except: pass
    # 2) per-point contributions (e.g., nll_i or resid^2)
    s=0.0; found=False
    if "nll" in R[0]:
        for row in R:
            try: s+=float(row["nll"]); found=True
            except: pass
        if found: return 2.0*s
    if "resid2" in R[0]:
        for row in R:
            try: s+=float(row["resid2"]); found=True
            except: pass
        if found: return s
    return None

def main():
    cands=list_candidates(prefix)
    if not cands:
        print("No candidates for", prefix, file=sys.stderr); sys.exit(2)
    chosen=None; chi2=None
    # prefer fit-like JSON/CSV/TXT next to prefix
    for path in reversed(cands):
        base=os.path.basename(path)
        if base.endswith(".normalized.json"): continue
        if base.endswith(".json"): 
            chi2=try_from_json(path)
        elif base.endswith(".csv"):
            chi2=try_from_csv(path)
        elif base.endswith(".txt") or base.endswith(".log"):
            # try lightweight grep-like read
            try:
                txt=open(path).read()
                import re as _re
                for rx,conv in [
                  (_re.compile(r"(?:chi2|chisq|chi[_ -]?square|cost|loss|rss|sse)\s*[:=]\s*([0-9eE+\-\.]+)", _re.I), float),
                  (_re.compile(r"(?:-?\s*2\s*(?:ln\s*L|logL)|neg2logL|minus2logL|minus2loglike)\s*[:=]\s*([0-9eE+\-\.]+)", _re.I), float),
                  (_re.compile(r"(?:nll|neg(?:ative)?\s*log(?:like|likelihood))\s*[:=]\s*([0-9eE+\-\.]+)", _re.I), lambda v:2.0*float(v)),
                  (_re.compile(r"(?:loglike|log[_ ]likelihood|lnL|logL)\s*[:=]\s*([0-9eE+\-\.]+)", _re.I), lambda v:-2.0*float(v)),
                ]:
                    m=rx.search(txt)
                    if m: chi2=conv(m.group(1)); break
            except Exception:
                pass
        if chi2 is not None: 
            chosen=path; break
    out = {"source": chosen or cands[-1], "chi2": chi2}
    outfn = re.sub(r"(\.json)?$", ".normalized.json", prefix)
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    json.dump(out, open(outfn,"w"), indent=2)
    print("WROTE", outfn, "chi2=", chi2, "from", out["source"])
if __name__=="__main__": main()
