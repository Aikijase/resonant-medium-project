import subprocess, json, os, re, sys, math, pathlib

BAO_DROP = "outputs/desi_bao_y_sigma.drop{idx}.csv"
COV_DROP = "outputs/bao_cov.drop{idx}.csv"
SN_VEC   = "outputs/sn_MATCHED_mu_zcut.csv"
SN_COV   = "outputs/Pantheon+SH0ES_STAT+SYS.cal.zcut.cov"
OUT      = "outputs/joint_LCDMplusNUIS_zcut_jk{idx}.json"
LOG      = "logs/jk/jk_lcdmnu_{idx}.log"

def run_and_capture(cmd, log_path):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pathlib.Path(log_path).write_text(p.stdout)
    return p.returncode, p.stdout

def ensure_json_written(out_path, stdout_text):
    # Some builds only print JSON; grab the last {...} block
    if os.path.exists(out_path):
        return True
    m = list(re.finditer(r'\{[\s\S]*\}', stdout_text))
    if not m:
        return False
    try:
        J = json.loads(m[-1].group(0))
    except Exception:
        return False
    pathlib.Path(out_path).write_text(json.dumps(J, indent=2))
    return True

def pinned_A_zero(json_path):
    if not os.path.exists(json_path): return False, "missing"
    J = json.load(open(json_path))
    fp = set(J.get("fit_params") or [])
    hb = J.get("hit_bounds") or {}
    bp = J.get("best_params") or {}
    # Accept if A not fitted OR exactly 0.0 OR pinned-at-bound 0.0
    if "A" not in fp: return True, f"A not in fit_params; bp.A={bp.get('A')}"
    try:
        Aval = float(bp.get("A", 1.0))
    except Exception:
        Aval = 1.0
    if Aval == 0.0: return True, "A=0.0"
    if hb.get("A") and Aval == 0.0: return True, "A pinned at bound 0.0"
    return False, f"fit_params={sorted(fp)}  A={bp.get('A')}  hit_bounds.A={hb.get('A')}"

def main():
    any_fail = False
    for i in range(12):
        in_csv = BAO_DROP.format(idx=i)
        in_cov = COV_DROP.format(idx=i)
        out    = OUT.format(idx=i)
        log    = LOG.format(idx=i)

        # Try 1: prior pin
        cmd1 = ["python3","joint_fit_resonant_anchors.py",
                "--bao-csv", in_csv, "--bao-cov", in_cov,
                "--sn-csv", SN_VEC, "--sn-cov", SN_COV,
                "--maxiter","5000", "--free-dmu",
                "--prior-A-sigma","0.0",
                "--out", out]
        rc, txt = run_and_capture(cmd1, log)
        ok = ensure_json_written(out, txt)
        ok2, msg = pinned_A_zero(out) if ok else (False, "no JSON")
        if ok2:
            print(f"[JK {i}] OK via prior pin  — {msg}")
            continue

        # Try 2: bounds pin (if supported by your build)
        cmd2 = ["python3","joint_fit_resonant_anchors.py",
                "--bao-csv", in_csv, "--bao-cov", in_cov,
                "--sn-csv", SN_VEC, "--sn-cov", SN_COV,
                "--maxiter","5000", "--free-dmu",
                "--bounds-A","0.0,0.0",
                "--out", out]
        rc, txt = run_and_capture(cmd2, log.replace(".log","_bounds.log"))
        ok = ensure_json_written(out, txt)
        ok2, msg = pinned_A_zero(out) if ok else (False, "no JSON")
        if ok2:
            print(f"[JK {i}] OK via bounds pin — {msg}")
            continue

        any_fail = True
        print(f"[JK {i}] FAIL to pin A=0 — see {log} and {log.replace('.log','_bounds.log')}")

    if any_fail:
        sys.exit(1)
    print("All LCDM+NUIS JK controls are pinned (A=0, dμ free).")

if __name__ == "__main__":
    main()
