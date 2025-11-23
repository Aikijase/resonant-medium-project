#!/usr/bin/env python3
"""
Run BAO (with local +5% sys at z=0.510) + fσ8 (viscous-corrected) + joint summary.

Defaults:
  - BAO: uses bao_measurements.csv, bao_covariance.csv → builds bao_covariance_adj.csv (+5% @ z=0.510 for DM & DH)
  - fσ8: uses fs8_measurements.csv; if no fs8_err, writes fs8_measurements_err05.csv with 5% fractional errors
  - k_eff list: 0.15, 0.20, 0.25 (you can change with --keff-list)

Outputs:
  - outputs/bao_fit_adj_LCDM.csv (+ meta)
  - outputs/fs8_keff_<k>_results.csv (+ plot + meta) for each k
  - outputs/joint_simple_summary.txt (printed summary)
"""

import os, sys, json, math, argparse, subprocess, shutil
import numpy as np
import pandas as pd
from pathlib import Path

C_KM_S = 299792.458

# ----------------- small utils -----------------
def run_cmd(cmd, env=None):
    print("[cmd]", " ".join(cmd))
    r = subprocess.run(cmd, env=env or os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(f"Command failed with code {r.returncode}: {' '.join(cmd)}")

def inv_with_jitter(C):
    eps = 1e-12 * float(np.median(np.diag(C)))
    Cf  = C + eps*np.eye(C.shape[0])
    try:
        from scipy.linalg import cho_factor, cho_solve
        cf = cho_factor(Cf, lower=True, check_finite=False)
        def Ci(v): return cho_solve(cf, v, check_finite=False)
    except Exception:
        Ci_mat = np.linalg.inv(Cf)
        def Ci(v): return Ci_mat @ v
    return Ci

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ----------------- BAO bits -----------------
def build_adj_cov(dump_csv, cov_in, cov_out, z_target=0.510, frac=0.05, kinds=None, tol=1e-6):
    """Add (frac*y_model)^2 to diagonal for rows at z≈z_target; optionally filter by kinds."""
    df = pd.read_csv(dump_csv)
    C  = np.loadtxt(cov_in, delimiter=",").astype(float)
    if C.shape[0] != len(df):
        raise SystemExit(f"[BAO cov] Cov shape {C.shape} != N {len(df)}")

    z = df["z"].to_numpy(float)
    ym = df["y_model"].to_numpy(float)
    kind = df["kind"].astype(str).str.upper().to_numpy()

    sel = np.isclose(z, z_target, atol=tol)
    if kinds:
        allowed = {k.strip().upper() for k in kinds.split(",")}
        sel = sel & np.isin(kind, list(allowed))

    idx = np.where(sel)[0]
    if len(idx) == 0:
        raise SystemExit(f"[BAO cov] No rows match z≈{z_target} and kinds={kinds or 'ALL'} in {dump_csv}")

    C2 = C.copy()
    for i in idx:
        C2[i,i] += (frac * ym[i])**2

    np.savetxt(cov_out, C2, delimiter=",")
    print(f"[bao-cov-adj] matched rows {idx.tolist()} at z≈{z_target}, kinds={kinds or 'ALL'}, frac={frac}")
    print(f"[bao-cov-adj] wrote {cov_out}")

def run_bao(bao_csv, cov_csv, out_root, Om0=0.3, h=0.7, stack_style="block"):
    run_cmd([
        sys.executable, "run_snbao_resonant.py",
        "--bao-csv", bao_csv,
        "--cov-csv", cov_csv,
        "--stack-style", stack_style,
        "--Om0", str(Om0), "--h", str(h),
        "--out-root", out_root
    ])
    dump = f"{out_root}_LCDM.csv"
    meta = f"{out_root}_LCDM_meta.json"
    if not Path(dump).exists() or not Path(meta).exists():
        raise SystemExit("[BAO] expected outputs not found")
    return dump, meta

def bao_stats(dump_csv, cov_csv):
    df = pd.read_csv(dump_csv)
    C  = np.loadtxt(cov_csv, delimiter=",").astype(float)
    y  = df["y_data"].to_numpy(float)
    ym = df["y_model"].to_numpy(float)
    Ci = inv_with_jitter(C)
    r  = y - ym
    chi2 = float(r @ Ci(r))
    dof  = max(len(y) - 1, 1)
    # r_d from beta uncertainty
    beta = float(json.load(open(dump_csv.replace(".csv","_meta.json")))["beta"])
    m_pre = ym / max(beta, 1e-300)
    den = float(m_pre @ Ci(m_pre))
    sigma_beta = 1.0 / (den**0.5)
    r_d = 1.0 / beta
    sigma_r_d = sigma_beta / (beta*beta)
    return dict(N=len(y), chi2=chi2, dof=dof, chi2_dof=chi2/dof,
                beta=beta, sigma_beta=sigma_beta, r_d=r_d, sigma_r_d=sigma_r_d)

# ----------------- fσ8 bits -----------------
def ensure_fs8_with_errors(src_csv, out_csv, frac=0.05, overwrite=False):
    df = pd.read_csv(src_csv)
    cols = {c.lower(): c for c in df.columns}
    zc   = cols.get("z") or cols.get("z_eff") or cols.get("redshift")
    fs8c = cols.get("fs8") or cols.get("fsigma8") or cols.get("fσ8") or cols.get("fs8_obs")
    errc = cols.get("fs8_err") or cols.get("sigma_fs8") or cols.get("err") or cols.get("uncertainty")
    if not zc or not fs8c:
        raise SystemExit(f"[fs8] need z and fs8-like columns. Got: {list(df.columns)}")

    out = df.copy()
    if errc is None or overwrite:
        out["fs8_err"] = pd.to_numeric(out[fs8c], errors="coerce") * float(frac)
        print(f"[fs8] wrote fs8_err = {frac:.3f} * fs8")
    else:
        out.rename(columns={errc: "fs8_err"}, inplace=True)
    out.rename(columns={zc: "z", fs8c: "fs8"}, inplace=True)
    out.to_csv(out_csv, index=False)
    print(f"[fs8] wrote {out_csv}")

def run_fs8(fs8_csv, dm_grid, keff, out_root):
    env = os.environ.copy()
    env["DM_GRID"] = str(dm_grid)
    env["DM_KEFF"] = str(keff)
    run_cmd([sys.executable, "fs8_pipeline.py", "--fs8-csv", fs8_csv, "--out-root", out_root], env=env)
    res = f"{out_root}_results.csv"
    if not Path(res).exists():
        raise SystemExit(f"[fs8] expected {res} not found")
    return res

def fs8_stats(results_csv):
    df = pd.read_csv(results_csv)
    low = {c.lower(): c for c in df.columns}
    c_obs = low.get("fs8_obs", low.get("fs8"))
    c_err = low.get("fs8_err") or low.get("sigma_fs8")
    c_th  = low.get("fs8_th_corr", low.get("fs8_th"))
    if not (c_obs and c_err and c_th):
        raise SystemExit(f"[fs8] missing needed columns in {results_csv}: {list(df.columns)}")
    obs = pd.to_numeric(df[c_obs], errors="coerce").to_numpy()
    err = pd.to_numeric(df[c_err], errors="coerce").to_numpy()
    th  = pd.to_numeric(df[c_th ], errors="coerce").to_numpy()
    m = np.isfinite(obs) & np.isfinite(err) & np.isfinite(th) & (err > 0)
    chi2 = float(((obs[m]-th[m])**2 / (err[m]**2)).sum())
    dof  = int(m.sum())  # no params profiled here
    return dict(N=dof, chi2=chi2, dof=dof, chi2_dof=(chi2/dof if dof>0 else np.nan))

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bao-csv", default="bao_measurements.csv")
    ap.add_argument("--bao-cov", default="bao_covariance.csv")
    ap.add_argument("--bao-cov-adj", default="bao_covariance_adj.csv")
    ap.add_argument("--bao-out-root", default="outputs/bao_fit_adj")
    ap.add_argument("--z-sys", type=float, default=0.510)
    ap.add_argument("--frac-sys", type=float, default=0.05)
    ap.add_argument("--fs8-csv", default="fs8_measurements.csv")
    ap.add_argument("--fs8-csv-err", default="fs8_measurements_err05.csv")
    ap.add_argument("--fs8-frac-err", type=float, default=0.05)
    ap.add_argument("--dm-grid", default="outputs/joint_visc_dm_growth_grid.csv")
    ap.add_argument("--keff-list", default="0.15,0.20,0.25")
    ap.add_argument("--Om0", type=float, default=0.3)
    ap.add_argument("--h",   type=float, default=0.7)
    ap.add_argument("--stack-style", default="block")
    ap.add_argument("--report", default="outputs/joint_simple_summary.txt")
    args = ap.parse_args()

    # 1) BAO w/ adjusted covariance
    # First, produce a temporary dump using the *original* cov to know which rows to tweak
    tmp_root = "outputs/_tmp_for_cov"
    dump0, meta0 = run_bao(args.bao_csv, args.bao_cov, tmp_root, args.Om0, args.h, args.stack_style)
    # Build adjusted covariance if not present
    if not Path(args.bao_cov_adj).exists():
        build_adj_cov(dump0, args.bao_cov, args.bao_cov_adj, z_target=args.z_sys, frac=args.frac_sys, kinds=None)
    # Now run BAO with adjusted covariance
    dump_adj, meta_adj = run_bao(args.bao_csv, args.bao_cov_adj, args.bao_out_root, args.Om0, args.h, args.stack_style)
    bao = bao_stats(dump_adj, args.bao_cov_adj)

    # 2) fσ8: ensure CSV with errors, then run for each k_eff
    ensure_fs8_with_errors(args.fs8_csv, args.fs8_csv_err, frac=args.fs8_frac_err, overwrite=False)
    keffs = [float(x) for x in args.keff_list.split(",")]
    fs8_rows = []
    for k in keffs:
        out_root = f"outputs/fs8_keff_{str(k).replace('.','_')}"
        res_csv = run_fs8(args.fs8_csv_err, args.dm_grid, k, out_root)
        stats = fs8_stats(res_csv)
        stats["keff"] = k
        stats["res"]  = res_csv
        fs8_rows.append(stats)
    fs8_df = pd.DataFrame(fs8_rows).sort_values("keff").reset_index(drop=True)

    # 3) Joint (naive sum of chi2 & dof; BAO already profiled β)
    report_lines = []
    report_lines.append("=== BAO (adjusted covariance) ===")
    report_lines.append(f"  N={bao['dof']+1}, chi2={bao['chi2']:.3f}, dof≈{bao['dof']}, chi2/dof={bao['chi2_dof']:.3f}")
    report_lines.append(f"  beta={bao['beta']:.8f} ± {bao['sigma_beta']:.8f}")
    report_lines.append(f"  r_d = {bao['r_d']:.2f} ± {bao['sigma_r_d']:.2f} Mpc")
    report_lines.append("")
    report_lines.append("=== fσ8 (per k_eff) ===")
    for r in fs8_rows:
        report_lines.append(f"  k_eff={r['keff']:.2f}: N={r['N']}, chi2={r['chi2']:.3f}, chi2/dof={r['chi2_dof']:.3f}")
    report_lines.append("")
    report_lines.append("=== JOINT (BAO + fσ8) ===")
    for r in fs8_rows:
        chi2_tot = bao["chi2"] + r["chi2"]
        dof_tot  = bao["dof"] + r["dof"]
        report_lines.append(f"  k_eff={r['keff']:.2f}: TOTAL chi2/dof = {chi2_tot:.3f}/{dof_tot} = {chi2_tot/dof_tot:.3f}")

    text = "\n".join(report_lines)
    print("\n" + text + "\n")
    ensure_dir(Path(args.report))
    with open(args.report, "w") as f:
        f.write(text)
    print(f"[report] wrote {args.report}")

if __name__ == "__main__":
    main()
