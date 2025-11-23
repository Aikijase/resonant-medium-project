#!/usr/bin/env python3
import argparse, json, os, subprocess, re, time, sys, pathlib

# Broad set of patterns to extract a chi-square–equivalent number from stdout.
# Order matters: the first match wins.
PATTERNS = [
    # direct chi2 / chisq / cost-like metrics on chi^2 scale
    (re.compile(r"(?:^|\s)(?:chi2|chisq|chi[_ -]?square|cost|loss|rss|sse)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda m: float(m.group(1))),
    # -2 log L variants (already chi^2 scale)
    (re.compile(r"(?:^|\s)(?:-?\s*2\s*(?:ln\s*L|logL)|neg2logL|minus2logL|minus2loglike)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda m: float(m.group(1))),
    # negative log-likelihood → chi2 = 2*NLL
    (re.compile(r"(?:^|\s)(?:nll|neg(?:ative)?\s*log(?:like|likelihood))\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda m: 2.0*float(m.group(1))),
    # plain log-likelihood → chi2 = -2*lnL
    (re.compile(r"(?:^|\s)(?:loglike|log[_ ]likelihood|lnL|logL)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
     lambda m: -2.0*float(m.group(1))),
    # "best ... =" style lines that include chi/lnL terms (very permissive fallback)
    (re.compile(r"best[^=\n]*=\s*([0-9eE+\-\.]+)\s*(?:$|\()"), lambda m: float(m.group(1))),
]

def extract_chi2(text: str):
    for rx, conv in PATTERNS:
        for m in rx.finditer(text):
            try:
                return conv(m)
            except Exception:
                pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config json for run_joint_bao_sn.py")
    ap.add_argument("--tag", default=None, help="optional short tag for the log filename")
    args = ap.parse_args()

    pathlib.Path("logs").mkdir(parents=True, exist_ok=True)

    ts  = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or pathlib.Path(args.config).stem
    log_path = f"logs/fit_{tag}_{ts}.log"

    # Run the launcher and capture all output
    proc = subprocess.Popen(
        ["python3", "run_joint_bao_sn.py", args.config],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    out_lines=[]
    with open(log_path, "w") as logf:
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            out_lines.append(line)
    code = proc.wait()

    text = "".join(out_lines)
    chi2 = extract_chi2(text)

    # Read config to pick output location
    with open(args.config) as f:
        C = json.load(f)
    out_prefix = C.get("out_prefix", "outputs/joint_run")
    out_json   = f"{out_prefix}.normalized.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    payload = {
        "config": C,
        "exit_code": code,
        "log_file": log_path,
        "chi2": chi2
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("WROTE", out_json, "chi2=", chi2, "exit_code=", code)

if __name__ == "__main__":
    main()
