import json, os, subprocess, sys, time, shlex

ROOT = os.getcwd()
SCRIPT = os.path.expanduser('~/sn_bao_joint_alpha_clean.py')

BAO_CSV = f"{ROOT}/data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV = f"{ROOT}/data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV  = f"{ROOT}/data/pantheon_plus/sn_MATCHED_mu.csv"

# Use the CSV version of the SN DIAG SPD to avoid text/binary decode issues
SN_DIAG_CSV = f"{ROOT}/data/pantheon_plus/Pantheon+SH0ES_DIAG.spd.csv"
if not os.path.exists(SN_DIAG_CSV):
    # auto-convert from .npy if needed
    npy = f"{ROOT}/data/pantheon_plus/Pantheon+SH0ES_DIAG.spd.npy"
    if os.path.exists(npy):
        import numpy as np, pandas as pd
        C = np.load(npy)
        pd.DataFrame(C).to_csv(SN_DIAG_CSV, index=False, header=False)

base_cmd = [
    sys.executable, "-u", SCRIPT,
    "--bao-csv", BAO_CSV,
    "--bao-cov", BAO_COV,
    "--sn-csv",  SN_CSV,
    "--sn-cov",  SN_DIAG_CSV,
    "--fid-H0", "70.0", "--fid-Om", "0.3", "--fid-rd", "147.1",
    "--rd-bounds", "147.1,147.1",
    "--prior-A-sigma", "0.4",
    "--f-bounds", "0.5,2.5", "--gamma-bounds", "0,1.0",
]

variants = [
    ["--fit", "A,f,phi,gamma"],
    ["--fit-params", "A,f,phi,gamma"],
    ["--model", "resonant"],
    ["--fit-A", "--fit-f", "--fit-phi", "--fit-gamma"],
    ["--resonant"],
    ["--do-fit"],
]

logdir = f"{ROOT}/logs"; os.makedirs(logdir, exist_ok=True)
outdir = f"{ROOT}/outputs"; os.makedirs(outdir, exist_ok=True)

def run_variant(extra, tag):
    out_json = f"{outdir}/joint_RESN_rd_fixed.{tag}.json"
    cmd = base_cmd + ["--out", out_json] + extra
    log = f"{logdir}/runner_{tag}.{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(log, "w") as lf:
        lf.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if not os.path.exists(out_json):
        return None, f"{tag}: no JSON produced (see {log})"
    try:
        with open(out_json) as f:
            d = json.load(f)
    except Exception as e:
        return None, f"{tag}: JSON read error: {e} (see {log})"
    return d, log

best = None
notes = []

for i, extra in enumerate(variants, 1):
    tag = f"v{i}"
    d, log = run_variant(extra, tag)
    if d is None:
        notes.append(log)
        continue
    fit_params = d.get("fit_params", [])
    chi2 = d.get("chi2", 1e99)
    ok = bool(fit_params) and chi2 < 1e3
    notes.append(f"{tag}: fit_params={fit_params}, chi2={chi2:.3f}, log={log}")
    if ok:
        best = (d, tag)
        break

print("\n=== Runner summary ===")
for n in notes:
    print(n)

if best is None:
    print("\nNo variant succeeded (fit_params stayed empty or chi2 too large). See logs above.")
    sys.exit(1)

# Write the winner to the canonical filename
winner, tag = best
final_json = f"{outdir}/joint_RESN_rd_fixed.json"
with open(final_json, "w") as f:
    json.dump(winner, f, indent=2)
print(f"\n✅ Success with {tag}. Copied to {final_json}")
print(f"fit_params={winner.get('fit_params')}, chi2={winner.get('chi2')}, ndof={winner.get('ndof')}")
bp = winner.get("best_params", {})
if bp:
    print("best:", {k: round(v, 6) for k, v in bp.items()})
