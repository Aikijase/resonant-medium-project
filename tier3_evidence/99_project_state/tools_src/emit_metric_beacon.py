#!/usr/bin/env python3
"""
Emit <prefix>.metric.txt with a line like:  chi2: <value>
Tries common keys in <prefix>.normalized.json then <prefix>.json,
and converts NLL/loglike if needed.
Usage:
  python3 tools/emit_metric_beacon.py outputs/phase8/joint_kappa0
"""
import json, sys, os, math
from pathlib import Path

def load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return None

def extract_chi2(J):
    if not isinstance(J, dict): return None
    # direct
    for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min","-2logL","neg2logL","minus2logL","minus2loglike","rss","sse"]:
        v = J.get(k)
        if isinstance(v,(int,float)): return float(v)
    # nested
    for sect in ["fit","result","summary","stats","metrics","breakdown"]:
        d = J.get(sect)
        if isinstance(d, dict):
            for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min","-2logL","neg2logL","minus2logL","minus2loglike","rss","sse"]:
                v = d.get(k)
                if isinstance(v,(int,float)): return float(v)
            # conversions
            for k in ["nll","negloglike","neg_loglike","neglogL","neg_logL"]:
                v = d.get(k)
                if isinstance(v,(int,float)): return 2.0*float(v)
            for k in ["loglike","log_likelihood","lnL","logL"]:
                v = d.get(k)
                if isinstance(v,(int,float)): return -2.0*float(v)
    # flat conversions
    for k in ["nll","negloglike","neg_loglike","neglogL","neg_logL"]:
        v = J.get(k)
        if isinstance(v,(int,float)): return 2.0*float(v)
    for k in ["loglike","log_likelihood","lnL","logL"]:
        v = J.get(k)
        if isinstance(v,(int,float)): return -2.0*float(v)
    return None

def main(stem):
    candidates = [f"{stem}.json", f"{stem}.normalized.json"]
    J = None
    for p in candidates:
        if os.path.exists(p):
            J = load_json(p)
            if J: break
    if not J:
        print(f"[emit_metric_beacon] no JSON neighbors for {stem}", file=sys.stderr)
        sys.exit(2)
    chi2 = extract_chi2(J)
    if chi2 is None:
        print(f"[emit_metric_beacon] couldn’t find a chi2-like key in {p}", file=sys.stderr)
        sys.exit(3)
    out = f"{stem}.metric.txt"
    with open(out, "w") as f:
        f.write(f"chi2: {chi2}\n")
    print("WROTE", out, "→ chi2:", chi2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tools/emit_metric_beacon.py <out_prefix_stem>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
