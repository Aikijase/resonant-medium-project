import os, re, sys, json, time, subprocess, shlex, tempfile

ROOT = os.getcwd()
SCRIPT_SRC = os.path.expanduser('~/sn_bao_joint_alpha_clean.py')

BAO_CSV = f"{ROOT}/data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV = f"{ROOT}/data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV  = f"{ROOT}/data/pantheon_plus/sn_MATCHED_mu.csv"
SN_COV  = f"{ROOT}/data/pantheon_plus/Pantheon+SH0ES_DIAG.spd.npy"  # we will patch loader to accept .npy

OUT_JSON = f"{ROOT}/outputs/joint_RESN_rd_fixed.json"
LOG_DIR  = f"{ROOT}/logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(f"{ROOT}/outputs", exist_ok=True)

def read_text(p):
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

src = read_text(SCRIPT_SRC)

# --- PATCH #1: robust covariance loader that handles .npy cleanly ---
loader_patch = r"""
def load_cov_matrix(path):
    import numpy as np, pandas as pd
    p = str(path).lower()
    if p.endswith('.npy'):
        C = np.load(path)
    else:
        try:
            C = pd.read_csv(path, header=None).values
        except Exception:
            C = np.loadtxt(path)
    C = 0.5*(C + C.T)
    return C
"""

if re.search(r"def\s+load_cov_matrix\s*\(", src):
    src = re.sub(r"def\s+load_cov_matrix\s*\([^)]*\):[\s\S]*?(?=\n\S)", loader_patch, src, count=1, flags=re.M)
else:
    src += "\n\n" + loader_patch + "\n"

# --- PATCH #2: force a fit of A,f,phi,gamma (defeat baseline mode) ---
# 2a) After args = parser.parse_args(), inject a force-fit block.
forcefit_block = r"""
# === Forced-fit block (injected) ===
try:
    _args = args
except NameError:
    pass
else:
    # Ensure we are not in baseline unless explicitly requested.
    is_baseline = getattr(_args, 'baseline', False)
    if not is_baseline:
        setattr(_args, 'fit_params', 'A,f,phi,gamma')
        # also set common toggles some scripts read
        for k in ('fit_A','fit_f','fit_phi','fit_gamma','do_fit','resonant'):
            setattr(_args, k, True)
        # model switch, if present
        if not hasattr(_args, 'model'):
            setattr(_args, 'model', 'resonant')
        elif getattr(_args, 'model', None) in (None, 'lcdm', 'baseline'):
            setattr(_args, 'model', 'resonant')
"""
if re.search(r"\n\s*args\s*=\s*parser\.parse_args\(\)\s*\n", src):
    src = re.sub(r"(\n\s*args\s*=\s*parser\.parse_args\(\)\s*\n)",
                 r"\1" + forcefit_block + "\n", src, count=1, flags=re.M)
else:
    # Fallback: append (some scripts bind args inside main)
    src += "\n\n" + forcefit_block + "\n"

# 2b) Hard override any explicit baseline assignments we can see.
# Replace occurrences like: fit_params = []
src = re.sub(r"\bfit_params\s*=\s*\[\s*\]", "fit_params = ['A','f','phi','gamma']", src)
# Replace string-typed empty: fit_params = ''
src = re.sub(r"\bfit_params\s*=\s*['\"]\s*['\"]", "fit_params = 'A,f,phi,gamma'", src)

# Write patched copy to a temp file
tmp_dir = tempfile.mkdtemp(prefix="resn_patch_")
SCRIPT_TMP = os.path.join(tmp_dir, "sn_bao_joint_alpha_clean.patched.py")
with open(SCRIPT_TMP, "w", encoding="utf-8") as f:
    f.write(src)

# Build/run command (rd basis, rd fixed)
cmd = [
    sys.executable, "-u", SCRIPT_TMP,
    "--bao-csv", BAO_CSV,
    "--bao-cov", BAO_COV,
    "--sn-csv",  SN_CSV,
    "--sn-cov",  SN_COV,
    "--fid-H0", "70.0", "--fid-Om", "0.3", "--fid-rd", "147.1",
    "--rd-bounds", "147.1,147.1",
    "--prior-A-sigma", "0.4",
    "--f-bounds", "0.5,2.5", "--gamma-bounds", "0,1.0",
    "--out", OUT_JSON
]
log_path = os.path.join(LOG_DIR, f"patch_and_run.{time.strftime('%Y%m%d-%H%M%S')}.log")
with open(log_path, "w") as lf:
    lf.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
    proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

# Summarize result
if not os.path.exists(OUT_JSON):
    print("❌ No JSON produced. See log:", log_path)
    print("patched_script:", SCRIPT_TMP)
    sys.exit(2)

d = json.load(open(OUT_JSON))
print("fit_params:", d.get("fit_params"))
print("chi2:", d.get("chi2"), "ndof:", d.get("ndof"))
print("breakdown:", d.get("breakdown"))
print("best_params:", d.get("best_params"))
print("log:", log_path)
print("patched_script:", SCRIPT_TMP)
