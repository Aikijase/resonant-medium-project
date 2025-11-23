import os, json, numpy as np, pandas as pd, subprocess as sp, math

RUNNER = "../run_joint_bao_sn.py"
BAO_VEC= "outputs/bao_long.csv"
BAO_COV= "data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV = "data/pantheon_plus/sn_MATCHED_mu.csv"
SN_COV = "outputs/Pantheon+SH0ES_STAT+SYS.cal.cov"
OUTCSV = "outputs/profile_fphi_k3.csv"
LOGDIR = "outputs/profile_logs"; os.makedirs(LOGDIR, exist_ok=True)

# 1) Load k=3 best-fit to center the grid
best = json.load(open("outputs/joint_RESN_k3_gamma0.json"))
f0   = float(best["best_params"]["f"])
phi0 = float(best["best_params"]["phi"])
A0   = float(best["best_params"]["A"])
def wrap_pi(x):
    # map to (-π, π]
    y = (x + math.pi) % (2*math.pi) - math.pi
    return y if y!= -math.pi else math.pi

# 2) Build a *focused* grid around (f0, phi0)
f_grid   = np.linspace(f0-0.06, f0+0.06, 25)   # ~0.54–0.66 if f0≈0.60
phi_grid = np.linspace(phi0-1.5, phi0+1.5, 61) # ~3π wide window
phi_grid = np.array([wrap_pi(x) for x in phi_grid])

def run_point(f, phi, try_with_A_bounds=False):
    out = "outputs/tmp_profile.json"
    cmd = ["python3", RUNNER,
           "--bao-csv", BAO_VEC, "--bao-cov", BAO_COV,
           "--sn-csv",  SN_CSV,  "--sn-cov",  SN_COV,
           "--model","resonant","--fit","A",
           "--f-bounds", f"{f},{f}",
           "--phi-bounds", f"{phi},{phi}",
           "--gamma-bounds","0,0",
           "--prior-A-sigma","0.4",
           "--out-json", out]
    if try_with_A_bounds:
        # gentle ±3σ box around A0 to help optimizer in tricky spots
        lo, hi = A0-1.2, A0+1.2
        cmd += ["--A-bounds", f"{lo},{hi}"]
    r = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    if r.returncode!=0 or not os.path.exists(out):
        tag = f"f{f:.5f}_phi{phi:.5f}"
        open(os.path.join(LOGDIR, f"fail_{tag}.log"), "w").write(r.stdout)
        return None
    try:
        j = json.load(open(out))
        return {"f":f, "phi":phi, "chi2":j["chi2"],
                "chi2_bao": j["breakdown"]["chi2_bao"],
                "chi2_sn":  j["breakdown"]["chi2_sn"]}
    except Exception as e:
        return None

rows=[]
for f in f_grid:
    for phi in phi_grid:
        rec = run_point(float(f), float(phi), try_with_A_bounds=False)
        if rec is None:
            rec = run_point(float(f), float(phi), try_with_A_bounds=True)
        if rec:
            rows.append(rec)
        else:
            # keep a placeholder so the grid shape is clear
            rows.append({"f":float(f), "phi":float(phi), "chi2":np.nan,
                         "chi2_bao":np.nan, "chi2_sn":np.nan})

df = pd.DataFrame(rows)
df.to_csv(OUTCSV, index=False)
print(f"[done] wrote {OUTCSV} with {df.notna().all(axis=1).sum()}/{len(df)} successful points; logs in {LOGDIR}/")
