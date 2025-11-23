#!/usr/bin/env python3
import argparse, os, json, numpy as np, pandas as pd, subprocess, sys, numpy.linalg as la
from pathlib import Path

def run(cmd, env=None):
    print("[cmd]", " ".join(cmd)); r = subprocess.run(cmd, env=env or os.environ.copy())
    if r.returncode != 0: raise SystemExit(f"failed: {' '.join(cmd)}")

def ensure_errors(fs8_csv, out_csv, frac):
    df = pd.read_csv(fs8_csv)
    low = {c.lower(): c for c in df.columns}
    zc   = low.get("z") or low.get("z_eff") or low.get("redshift")
    fs8c = low.get("fs8") or low.get("fsigma8") or low.get("fσ8") or low.get("fs8_obs")
    errc = low.get("fs8_err") or low.get("sigma_fs8") or low.get("err") or low.get("uncertainty")
    if not zc or not fs8c: raise SystemExit(f"[fs8] need z & fs8-like columns. Got {list(df.columns)}")
    out = df.copy()
    if errc is None:
        out["fs8_err"] = pd.to_numeric(out[fs8c], errors="coerce") * float(frac)
        print(f"[fs8] created fs8_err = {frac:.3f} * fs8")
    else:
        out.rename(columns={errc: "fs8_err"}, inplace=True)
    out.rename(columns={zc: "z", fs8c: "fs8"}, inplace=True)
    out.to_csv(out_csv, index=False)
    return out_csv

def fs8_chi2(results_csv):
    df = pd.read_csv(results_csv)
    low = {c.lower(): c for c in df.columns}
    obs = pd.to_numeric(df[low.get("fs8_obs", low.get("fs8"))], errors="coerce").to_numpy()
    err = pd.to_numeric(df[low.get("fs8_err", "sigma_fs8")], errors="coerce").to_numpy()
    th  = pd.to_numeric(df[low.get("fs8_th_corr", low.get("fs8_th"))], errors="coerce").to_numpy()
    m = np.isfinite(obs) & np.isfinite(err) & np.isfinite(th) & (err>0)
    N = int(m.sum())
    chi2 = float(((obs[m]-th[m])**2/(err[m]**2)).sum())
    return chi2, N

def bao_chi2(dump_csv, cov_csv):
    df = pd.read_csv(dump_csv)
    C  = np.loadtxt(cov_csv, delimiter=",").astype(float)
    r  = df["y_data"].to_numpy(float) - df["y_model"].to_numpy(float)
    Ci = la.inv(C)
    chi2 = float(r @ (Ci @ r))
    dof  = len(df) - 1
    return chi2, dof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nus", default="0.0,0.2,0.5,0.8,1.0")
    ap.add_argument("--keff", type=float, default=0.20)
    ap.add_argument("--om0",  type=float, default=0.3)
    ap.add_argument("--h",    type=float, default=0.7)
    ap.add_argument("--fs8",  default="fs8_measurements.csv")
    ap.add_argument("--fs8-out", default="fs8_measurements_err05.csv")
    ap.add_argument("--err-frac", type=float, default=0.05)
    ap.add_argument("--bao-dump", default="outputs/bao_fit_adj_LCDM.csv")
    ap.add_argument("--bao-cov",  default="bao_covariance_adj.csv")
    ap.add_argument("--out",   default="outputs/fs8_nu_sweep_summary.csv")
    args = ap.parse_args()

    fs8_use = ensure_errors(args.fs8, args.fs8_out, args.err_frac)

    rows=[]
    for s in args.nus.split(","):
        nu = float(s)
        root = f"outputs/visc_nu_{nu}"
        # growth grid
        run([sys.executable, "viscous_dm_growth.py",
             "--nu0", str(nu), "--cs2", "0.0",
             "--Om0", str(args.om0), "--h", str(args.h),
             "--k-list", "0.05,0.10,0.20,0.30",
             "--a-min", "1e-3", "--n-a", "800",
             "--out-root", root])
        grid = f"{root}_growth_grid.csv"
        if not Path(grid).exists(): raise SystemExit(f"missing grid {grid}")

        # fs8 run
        env = os.environ.copy()
        env["DM_GRID"] = grid
        env["DM_KEFF"] = str(args.keff)
        out_root = f"outputs/fs8_nu_{nu}"
        run([sys.executable, "fs8_pipeline.py", "--fs8-csv", fs8_use, "--out-root", out_root], env=env)
        res = f"{out_root}_results.csv"
        chi2, N = fs8_chi2(res)
        rows.append(dict(nu0=nu, keff=args.keff, N=N, chi2=chi2, chi2_dof=(chi2/N if N>0 else np.nan), res=res, grid=grid))

    df = pd.DataFrame(rows).sort_values(["keff","nu0"]).reset_index(drop=True)

    # optional joint with BAO (if files exist)
    if Path(args.bao_dump).exists() and Path(args.bao_cov).exists():
        chi2_bao, dof_bao = bao_chi2(args.bao_dump, args.bao_cov)
        df["chi2_bao"] = chi2_bao
        df["dof_bao"]  = dof_bao
        df["chi2_joint"] = df["chi2_bao"] + df["chi2"]
        df["dof_joint"]  = df["dof_bao"] + df["N"]
        df["chi2_dof_joint"] = df["chi2_joint"] / df["dof_joint"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    with pd.option_context("display.max_rows", 100, "display.width", 140):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
if __name__ == "__main__":
    main()
