#!/usr/bin/env python3
"""
Run the joint fit via run_joint_bao_sn.py, then emit a stable metrics JSON:
    <out_prefix>.fit.json    ->  { "chi2": <float or null>, "source": "<path or 'stdout'>", ... }

Usage:
    python3 tools/make_fit_json.py configs/baseline.json [--tag baseline]

Notes:
- We DO NOT modify the fitter; we just orchestrate, capture logs, and harvest metrics
  from files next to out_prefix (preferred) or from stdout as a fallback.
- Works with your existing configs that set "out_prefix".
"""
import argparse, json, os, sys, subprocess, time, shlex, glob, re, csv
from pathlib import Path

# ---------- metric extraction helpers ----------

# 1) From stdout text
_PATTERNS = [
    # direct chi2 / chisq / cost-like on chi^2 scale
    (re.compile(r"(?:^|\s)(?:chi2|chisq|chi[_ -]?square|cost|loss|rss|sse)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda v: float(v)),
    # -2 log L variants (already chi^2-scale)
    (re.compile(r"(?:^|\s)(?:-?\s*2\s*(?:ln\s*L|logL)|neg2logL|minus2logL|minus2loglike)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda v: float(v)),
    # NLL → chi2 = 2*NLL
    (re.compile(r"(?:^|\s)(?:nll|neg(?:ative)?\s*log(?:like|likelihood))\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda v: 2.0*float(v)),
    # loglike → chi2 = -2*lnL
    (re.compile(r"(?:^|\s)(?:loglike|log[_ ]likelihood|lnL|logL)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda v: -2.0*float(v)),
]

def chi2_from_text(txt: str):
    for rx, conv in _PATTERNS:
        m = rx.search(txt)
        if m:
            try: return conv(m.group(1))
            except Exception: pass
    return None

# 2) From JSON
def chi2_from_json(path: str):
    try:
        J = json.load(open(path))
    except Exception:
        return None, None
    # direct
    for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min","-2logL","neg2logL","minus2logL","minus2loglike"]:
        v = J.get(k)
        if isinstance(v,(int,float)): return float(v), "json:"+path
    # nested
    for sect in ["fit","result","summary","stats","metrics"]:
        d = J.get(sect)
        if isinstance(d, dict):
            for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","chi2_min","-2logL","neg2logL","minus2logL","minus2loglike"]:
                v = d.get(k)
                if isinstance(v,(int,float)): return float(v), "json:"+path
            # NLL / loglike conversions
            for k in ["nll","negloglike","neg_loglike","neglogL","neg_logL"]:
                v = d.get(k)
                if isinstance(v,(int,float)): return 2.0*float(v), "json:"+path
            for k in ["loglike","log_likelihood","lnL","logL"]:
                v = d.get(k)
                if isinstance(v,(int,float)): return -2.0*float(v), "json:"+path
    # flat conversions
    for k in ["nll","negloglike","neg_loglike","neglogL","neg_logL"]:
        v = J.get(k)
        if isinstance(v,(int,float)): return 2.0*float(v), "json:"+path
    for k in ["loglike","log_likelihood","lnL","logL"]:
        v = J.get(k)
        if isinstance(v,(int,float)): return -2.0*float(v), "json:"+path
    return None, None

# 3) From CSV (single-row metrics or per-point contributions)
def chi2_from_csv(path: str):
    try:
        rows = list(csv.DictReader(open(path, newline="")))
    except Exception:
        return None, None
    if not rows: return None, None
    if len(rows) == 1:
        r = rows[0]
        for k in ["chi2","chisq","chi_sq","chi2_total","best_chi2","-2logL","neg2logL","minus2logL","minus2loglike"]:
            if k in r:
                try: return float(r[k]), "csv:"+path
                except: pass
        for k in ["nll","negloglike","neg_loglike","neglogL","neg_logL"]:
            if k in r:
                try: return 2.0*float(r[k]), "csv:"+path
                except: pass
        for k in ["loglike","log_likelihood","lnL","logL"]:
            if k in r:
                try: return -2.0*float(r[k]), "csv:"+path
                except: pass
    # per-point contributions
    s = 0.0; seen = False
    if "nll" in rows[0]:
        for r in rows:
            try: s += float(r["nll"]); seen = True
            except: pass
        if seen: return 2.0*s, "csv-sum-nll:"+path
    if "resid2" in rows[0]:
        for r in rows:
            try: s += float(r["resid2"]); seen = True
            except: pass
        if seen: return s, "csv-sum-resid2:"+path
    return None, None

# ---------- main orchestration ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to JSON config for run_joint_bao_sn.py")
    ap.add_argument("--tag", default=None, help="optional tag for log filename")
    args = ap.parse_args()

    C = json.load(open(args.config))
    out_prefix = C.get("out_prefix", "outputs/joint_run")
    Path("logs").mkdir(parents=True, exist_ok=True)

    ts  = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or Path(args.config).stem
    log_path = f"logs/make_fit_{tag}_{ts}.log"

    # 1) Run the launcher and capture all output
    proc = subprocess.Popen(
        ["python3", "run_joint_bao_sn.py", args.config],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    lines = []
    with open(log_path, "w") as lf:
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            lines.append(line)
    code = proc.wait()
    text = "".join(lines)

    # 2) Prefer harvesting from files next to out_prefix
    deny = re.compile(r"(normalized|zero_pad_sweep|sweep_dt\d|mt_|surrogate_|loo_summary|cov_study|mock_delta_chi2|scan_penalty)")
    neigh = sorted([p for p in glob.glob(out_prefix+"*") if not p.endswith(".normalized.json")], key=os.path.getmtime)
    neigh = [p for p in neigh if not deny.search(os.path.basename(p))]

    chi2, source = None, None
    # try JSONs then CSVs then TXT/LOG
    for p in reversed(neigh):
        if p.endswith(".json"):
            chi2, source = chi2_from_json(p)
        elif p.endswith(".csv"):
            chi2, source = chi2_from_csv(p)
        elif p.endswith(".txt") or p.endswith(".log"):
            try:
                chi2_candidate = chi2_from_text(open(p).read())
                if chi2_candidate is not None:
                    chi2, source = chi2_candidate, "txt:"+p
            except Exception:
                pass
        if chi2 is not None: break

    # 3) Fallback to stdout scrape if no file worked
    if chi2 is None:
        chi2 = chi2_from_text(text)
        if chi2 is not None:
            source = "stdout"

    # 4) Write a stable fit JSON
    out_fit = f"{out_prefix}.fit.json"
    Path(os.path.dirname(out_fit)).mkdir(parents=True, exist_ok=True)
    payload = {
        "config": C,
        "exit_code": code,
        "log_file": log_path,
        "chi2": chi2,
        "source": source
    }
    json.dump(payload, open(out_fit, "w"), indent=2)
    print("WROTE", out_fit, "chi2=", chi2, "source=", source, "exit_code=", code)

if __name__ == "__main__":
    main()
