#!/usr/bin/env python3
import sys, json, subprocess as sp, shlex, os

"""
Usage:
  python3 run_joint_bao_sn.py run_cfg.json
"""

def run(cmd):
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    sp.run(cmd, check=True)

def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".json"):
        print("Usage: python3 run_joint_bao_sn.py <config.json>")
        sys.exit(2)
    with open(sys.argv[1]) as f:
        C = json.load(f)

    ROOT = os.path.abspath(C.get("root", "."))

    # Required I/O
    bao_csv = os.path.join(ROOT, C["bao_csv"])
    bao_cov = os.path.join(ROOT, C["bao_cov"])
    sn_csv  = os.path.join(ROOT, C["sn_csv"])
    sn_cov  = os.path.join(ROOT, C["sn_cov"])
    out_prefix = os.path.join(ROOT, C.get("out_prefix", "outputs/joint_run"))

    # Optional params with sane defaults
    H0   = str(C.get("H0", 70.0))
    Om   = str(C.get("Om", 0.3))
    rd   = str(C.get("rd", 147.1))
    Ok   = str(C.get("Ok", 0.0))
    Or   = str(C.get("Or", 0.0))
    prior_A_sigma = str(C.get("prior_A_sigma", 1.0))     # if your guard uses it
    k_params      = str(C.get("k_params", 5))            # total free params if needed
    assume_per_rd = C.get("assume_per_rd", True)         # DESI DR1 long vector is /rd basis

    # Build command to your current guarded runner
    cmd = [
        "python3", os.path.join(ROOT, "joint_guard.py"),
        "--bao-csv", bao_csv,
        "--bao-cov", bao_cov,
        "--sn-csv",  sn_csv,
        "--sn-cov",  sn_cov,
        "--out-prefix", out_prefix,
        "--H0", H0, "--Om", Om, "--Or", Or, "--Ok", Ok, "--rd", rd,
        "--prior-A-sigma", prior_A_sigma,
        "--k-params", k_params,
    ]
    if assume_per_rd:
        cmd.append("--assume-per-rd")

    # Optional toggles you might have in joint_guard.py
    if C.get("plus_lya", True):        # keep DR1+Lyα unless you want to drop it
        cmd.append("--plus-lya")
    if C.get("diag_only", False):
        cmd.append("--diag-only")

    run(cmd)

if __name__ == "__main__":
    main()
