import subprocess, json, os, re, pathlib, sys

BAO_CSV = "outputs/desi_bao_y_sigma.drop{idx}.csv"
BAO_COV = "outputs/bao_cov.drop{idx}.csv"
SN_CSV  = "outputs/sn_MATCHED_mu_zcut.csv"
SN_COV  = "outputs/Pantheon+SH0ES_STAT+SYS.cal.zcut.cov"
OUT     = "outputs/joint_LCDMplusNUIS_zcut_jk{idx}.json"

def run(cmd, logpath):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pathlib.Path(logpath).write_text(p.stdout)
    return p.returncode, p.stdout

def ensure_json(outpath, stdout_text):
    if os.path.exists(outpath):
        return True
    m = list(re.finditer(r'\{[\s\S]*\}\s*$', stdout_text))
    if not m: return False
    try:
        J = json.loads(m[-1].group(0))
    except Exception:
        return False
    pathlib.Path(outpath).write_text(json.dumps(J, indent=2))
    return True

def pinned_ok(path):
    if not os.path.exists(path): return False, "missing"
    J = json.load(open(path))
    fp = set(J.get("fit_params") or [])
    hb = J.get("hit_bounds") or {}
    bp = J.get("best_params") or {}
    Aval = float(bp.get("A", 1.0)) if bp.get("A") is not None else 1.0
    if "A" not in fp: return True, f"A not fit; A={Aval}"
    if Aval == 0.0:   return True, "A=0.0"
    if hb.get("A") and Aval == 0.0: return True, "A pinned at bound 0.0"
    return False, f"fit_params={sorted(fp)} A={bp.get('A')} hit_bounds.A={hb.get('A')}"

def build_cmd(base, variant):
    return base + variant

def main():
    any_fail = False
    for i in range(12):
        out = OUT.format(idx=i)
        csv = BAO_CSV.format(idx=i)
        cov = BAO_COV.format(idx=i)
        base = ["python3","joint_fit_resonant_anchors.py",
                "--bao-csv", csv, "--bao-cov", cov,
                "--sn-csv", SN_CSV, "--sn-cov", SN_COV,
                "--maxiter","5000", "--free-dmu",
                "--out", out]
        variants = [
            ["--A-bounds","0.0,0.0"],
            ["--A_bounds","0.0,0.0"],
            ["--prior-A-sigma","0.0"],
            ["--prior_A_sigma","0.0"],
            ["--A-bounds","0.0,0.0","--prior-A-sigma","0.0"],
            ["--A_bounds","0.0,0.0","--prior_A_sigma","0.0"],
            # max clamp fallback disables the resonance sector entirely:
            ["--A-bounds","0.0,0.0","--f-bounds","1.0,1.0","--gamma-bounds","0.0,0.0","--phi-bounds","0.0,0.0"],
            ["--A_bounds","0.0,0.0","--f_bounds","1.0,1.0","--gamma_bounds","0.0,0.0","--phi_bounds","0.0,0.0"],
        ]
        ok=False; msg="(not tried)"
        for vi, v in enumerate(variants):
            log = f"logs/jk/jk_lcdmnu_{i}_v{vi}.log"
            rc, outtxt = run(build_cmd(base, v), log)
            if not ensure_json(out, outtxt): 
                continue
            ok, msg = pinned_ok(out)
            if ok:
                print(f"[JK {i}] OK via variant {vi}: {msg}")
                break
        if not ok:
            any_fail = True
            print(f"[JK {i}] FAIL to pin A=0  (last: {msg})  — check logs/jk/jk_lcdmnu_{i}_v*.log")
    if any_fail:
        sys.exit(1)
    print("All pinned JK controls written.")
if __name__ == "__main__":
    main()
