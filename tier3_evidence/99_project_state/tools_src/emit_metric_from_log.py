#!/usr/bin/env python3
"""
Read <stem>.normalized.json (or <stem>.json), open its "log_file", scrape a chi^2-like
metric, and emit <stem>.metric.txt as:  chi2: <number>

Usage:
  python3 tools/emit_metric_from_log.py outputs/phase8/joint_kappa0
"""
from __future__ import annotations
import sys, os, json, re
from pathlib import Path

# Regexes: chi2 / -2lnL / NLL / loglike (converted to chi2 scale where needed)
RX = [
    (re.compile(r"(?:^|\s)(?:chi\s*\^?\s*2|chi2|chisq|χ\s*2|χ²|chi[_ -]?square)\s*[:=]\s*([0-9eE+\-\.]+)", re.I), lambda v: float(v)),
    (re.compile(r"(?:^|\s)(?:-?\s*2\s*(?:ln\s*L|logL)|neg2logL|minus2logL|minus2loglike)\s*[:=]\s*([0-9eE+\-\.]+)", re.I), lambda v: float(v)),
    (re.compile(r"(?:^|\s)(?:nll|neg(?:ative)?\s*log(?:like|likelihood))\s*[:=]\s*([0-9eE+\-\.]+)", re.I), lambda v: 2.0*float(v)),
    (re.compile(r"(?:^|\s)(?:loglike|log[_ ]likelihood|lnL|logL)\s*[:=]\s*([0-9eE+\-\.]+)", re.I), lambda v: -2.0*float(v)),
]

def scrape_chi2(text: str):
    for rx, conv in RX:
        m = rx.search(text)
        if m:
            try:
                x = conv(m.group(1))
                if x == x and x not in (float("inf"), float("-inf")):
                    return x
            except Exception:
                pass
    return None

def load_json_first(stem: str):
    for suf in (".normalized.json", ".json"):
        p = f"{stem}{suf}"
        if os.path.exists(p):
            try:
                return json.load(open(p)), p
            except Exception:
                pass
    return None, None

def main(stem: str):
    J, jpath = load_json_first(stem)
    if not isinstance(J, dict):
        print(f"[emit_metric_from_log] no JSON neighbor at {stem}.normalized.json/.json", file=sys.stderr)
        return 2
    log = J.get("log_file")
    if not (isinstance(log, str) and os.path.exists(log)):
        print(f"[emit_metric_from_log] JSON has no usable 'log_file' (json={jpath})", file=sys.stderr)
        return 3
    txt = open(log, "r", errors="ignore").read()
    chi2 = scrape_chi2(txt)
    if chi2 is None:
        # show a hint so you can grep manually
        print(f"[emit_metric_from_log] couldn't scrape χ² from log. Try grep:", file=sys.stderr)
        print(f"  grep -Ei 'chi.?2|chisq|minus2log|neg2log|nll|loglike|lnL' {log}", file=sys.stderr)
        return 4
    out = f"{stem}.metric.txt"
    with open(out, "w") as f:
        f.write(f"chi2: {chi2}\n")
    print("WROTE", out, "←", log, "→ chi2:", chi2)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tools/emit_metric_from_log.py <out_prefix_stem>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
