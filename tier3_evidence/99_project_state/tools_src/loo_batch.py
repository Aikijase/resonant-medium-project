import os, json, math, subprocess as sp
import numpy as np, pandas as pd

ROOT   = os.getcwd()
RUNNER = os.path.abspath(os.path.join(ROOT, "..", "run_joint_bao_sn.py"))
BAO_VEC= "outputs/bao_long.csv"
BAO_COV= "data/desi_dr1_bao/bao_covariance_plus_lya.csv"
SN_CSV = "data/pantheon_plus/sn_MATCHED_mu.csv"
SN_COV = "outputs/Pantheon+SH0ES_STAT+SYS.cal.cov"
OUT    = "outputs/loo"
os.makedirs(OUT, exist_ok=True)

for p in [RUNNER, BAO_VEC, BAO_COV, SN_CSV, SN_COV]:
    if not os.path.exists(p):
        raise SystemExit(f"[missing] {p}")

v = pd.read_csv(BAO_VEC)
C = np.loadtxt(BAO_COV, delimiter=",")
if not (len(v)==C.shape[0]==C.shape[1]):
    raise SystemExit(f"[mismatch] BAO vec={len(v)}  C={C.shape}")

rows=[]
for i in range(len(v)):
    print(f"\n=== LOO i={i} ===", flush=True)
    vec=f"{OUT}/bao_long_minus_{i}.csv"
    cov=f"{OUT}/bao_cov_minus_{i}.csv"
    v.drop(v.index[i]).reset_index(drop=True).to_csv(vec, index=False)
    np.savetxt(cov, np.delete(np.delete(C,i,0),i,1), fmt="%.8e", delimiter=",")

    def run(tag, args):
        jout=f"{OUT}/{tag}.json"; log=f"{OUT}/{tag}.log"
        cmd=["python3", RUNNER,
             "--bao-csv", vec, "--bao-cov", cov,
             "--sn-csv",  SN_CSV, "--sn-cov", SN_COV,
             "--model"] + args + ["--out-json", jout]
        r=sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, cwd=ROOT)
        open(log,"w").write(r.stdout)
        if r.returncode or not os.path.exists(jout):
            print(f"[FAIL] {tag}. See {log}")
            raise SystemExit(1)
        j=json.load(open(jout))
        print(f"[{tag}] chi2_tot={j['chi2']:.3f}  bao={j['breakdown']['chi2_bao']:.3f}  sn={j['breakdown']['chi2_sn']:.3f}", flush=True)
        return j

    jl=run(f"lcdm_i{i}", ["lcdm"])
    jr=run(f"resn_i{i}", ["resonant","--fit","A","f","phi","gamma"])

    N=jl["ic"]["n_points"]; k0=len(jl.get("fit_params",[])); k1=len(jr.get("fit_params",[]))
    dchi=jr["chi2"]-jl["chi2"]; dAIC=dchi+2*(k1-k0); dBIC=dchi+(k1-k0)*math.log(N)
    rows.append({"i":i,"N_bao":len(v)-1,
                 "chi2_bao_lcdm":jl["breakdown"]["chi2_bao"],
                 "chi2_bao_resn":jr["breakdown"]["chi2_bao"],
                 "chi2_sn":jl["breakdown"]["chi2_sn"],
                 "chi2_lcdm":jl["chi2"],"chi2_resn":jr["chi2"],
                 "dchi":dchi,"dAIC":dAIC,"dBIC":dBIC})
    print(f"[i={i}] Δχ²={dchi:.3f}  ΔAIC={dAIC:.3f}  ΔBIC={dBIC:.3f}", flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/loo_summary.csv", index=False)
print(f"\n[OK] wrote {OUT}/loo_summary.csv (N={len(rows)})", flush=True)
