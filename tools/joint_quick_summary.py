#!/usr/bin/env python3
import argparse, json, os
import numpy as np, pandas as pd

def chi2_bao(dump_csv, cov_csv):
    df = pd.read_csv(dump_csv)
    C  = np.loadtxt(cov_csv, delimiter=",").astype(float)
    y  = df["y_data"].to_numpy(float)
    ym = df["y_model"].to_numpy(float)
    r  = y - ym
    if C.shape[0] != len(r):
        raise SystemExit(f"[BAO] Cov size {C.shape} != N {len(r)} — stacking mismatch?")
    Ci = np.linalg.inv(C)
    chi2 = float(r @ (Ci @ r))
    N = len(r)
    k = 1  # profiled β in the BAO fit
    dof = max(N - k, 1)
    return {"chi2": chi2, "N": N, "dof": dof, "red": chi2/dof}

def chi2_fs8(results_csv):
    df = pd.read_csv(results_csv)
    if not {"fs8_obs","fs8_th","fs8_err"}.issubset(df.columns):
        # Old columns? try to infer
        raise SystemExit(f"[fσ8] Unexpected columns in {results_csv}: {list(df.columns)}")
    obs  = df["fs8_obs"].to_numpy(float)
    th   = df["fs8_th"].to_numpy(float)
    err  = df["fs8_err"].to_numpy(float)
    m = np.isfinite(err) & (err > 0)
    n = int(m.sum())
    if n == 0:
        return {"chi2": float("nan"), "N": 0, "dof": 0, "red": float("nan")}
    chi2 = float(((obs[m]-th[m])**2 / (err[m]**2)).sum())
    # Rough dof: we didn't refit cosmology here, so just count data points
    dof = n  # use n (or n - p for a stricter take)
    return {"chi2": chi2, "N": n, "dof": dof, "red": chi2/dof}

def main():
    ap = argparse.ArgumentParser(description="Quick joint summary: BAO + fσ8.")
    ap.add_argument("--bao-dump", default="outputs/bao_fit_LCDM.csv")
    ap.add_argument("--bao-cov",  default="bao_covariance.csv")
    ap.add_argument("--fs8-results", default=None, help="Optional fs8 results CSV (e.g. outputs/fs8_visc_corr_results.csv)")
    ap.add_argument("--out-root", default="outputs/joint_quick_summary")
    args = ap.parse_args()

    out = {}
    out["bao"] = chi2_bao(args.bao_dump, args.bao_cov)

    if args.fs8_results and os.path.exists(args.fs8_results):
        out["fs8"] = chi2_fs8(args.fs8_results)
    else:
        out["fs8"] = {"chi2": float("nan"), "N": 0, "dof": 0, "red": float("nan")}

    # naive combined (note: parameters shared across datasets are ignored here)
    chi2_tot = 0.0
    dof_tot  = 0
    for sect in ("bao","fs8"):
        if np.isfinite(out[sect]["chi2"]) and out[sect]["dof"] > 0:
            chi2_tot += out[sect]["chi2"]
            dof_tot  += out[sect]["dof"]
    out["total"] = {
        "chi2": chi2_tot if dof_tot>0 else float("nan"),
        "dof":  dof_tot,
        "red":  (chi2_tot/dof_tot) if dof_tot>0 else float("nan")
    }

    os.makedirs(os.path.dirname(args.out_root), exist_ok=True) if os.path.dirname(args.out_root) else None
    with open(args.out_root + ".json", "w") as f:
        json.dump(out, f, indent=2)

    # pretty print
    print("BAO : χ²={chi2:.3f}, dof={dof}, χ²/dof={red:.3f}, N={N}".format(**out["bao"]))
    if out["fs8"]["N"]>0:
        print("fσ8: χ²={chi2:.3f}, dof={dof}, χ²/dof={red:.3f}, N={N}".format(**out["fs8"]))
    else:
        print("fσ8: (no finite error bars → skipped χ²)")
    if out["total"]["dof"]>0:
        print("TOTAL: χ²={chi2:.3f}, dof={dof}, χ²/dof={red:.3f}".format(**out["total"]))
