#!/usr/bin/env python3
# fs8_pipeline.py
# Standalone fσ8 pipeline with optional viscous-DM correction via a precomputed grid.

import os, csv, math, argparse, json
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- LCDM background + growth ----------
MPC_IN_KM = 3.0856775814913673e19
SEC_PER_YEAR = 3.15576e7
SEC_PER_GYR  = SEC_PER_YEAR * 1.0e9

def H0_to_per_Gyr(H0_km_s_Mpc: float) -> float:
    H0_per_s = H0_km_s_Mpc / MPC_IN_KM
    return H0_per_s * SEC_PER_GYR

def E_of_a(a, Om0, Ode0):
    return math.sqrt(Om0*a**(-3) + Ode0)

def H_of_a_per_Gyr(a, H0_per_Gyr, Om0, Ode0):
    return H0_per_Gyr * E_of_a(a, Om0, Ode0)

def Omega_m_of_a(a, Om0, Ode0):
    return (Om0*a**(-3)) / (Om0*a**(-3) + Ode0)

def dlnH_dN(a, Om0, Ode0):
    return -1.5 * Omega_m_of_a(a, Om0, Ode0)

def solve_growth_LCDM(a_grid, H0_per_Gyr, Om0, Ode0):
    N = np.log(a_grid)
    D  = np.zeros_like(N)
    Dp = np.zeros_like(N)
    D[0]  = a_grid[0]
    Dp[0] = 1.0
    def A(a): return 2.0 + dlnH_dN(a, Om0, Ode0)
    def B(a): return -1.5 * Omega_m_of_a(a, Om0, Ode0)
    for i in range(len(N)-1):
        a  = a_grid[i]
        hN = N[i+1] - N[i]
        def deriv(a_local, D_local, Dp_local):
            Dpp = -A(a_local)*Dp_local - B(a_local)*D_local
            return Dp_local, Dpp
        # RK4 in N
        k1_Dp, k1_Dpp = deriv(a, D[i], Dp[i])
        a2 = np.exp(N[i] + 0.5*hN)
        k2_Dp, k2_Dpp = deriv(a2, D[i] + 0.5*hN*k1_Dp, Dp[i] + 0.5*hN*k1_Dpp)
        k3_Dp, k3_Dpp = deriv(a2, D[i] + 0.5*hN*k2_Dp, Dp[i] + 0.5*hN*k2_Dpp)
        a4 = np.exp(N[i] + hN)
        k4_Dp, k4_Dpp = deriv(a4, D[i] + hN*k3_Dp, Dp[i] + hN*k3_Dpp)
        D[i+1]  = D[i]  + (hN/6.0)*(k1_Dp + 2*k2_Dp + 2*k3_Dp + k4_Dp)
        Dp[i+1] = Dp[i] + (hN/6.0)*(k1_Dpp+ 2*k2_Dpp+ 2*k3_Dpp+ k4_Dpp)
    return D / D[-1]

def growth_rate_f(D: np.ndarray, a: np.ndarray) -> np.ndarray:
    f = np.zeros_like(D)
    # central difference in ln a
    for i in range(len(a)):
        if i == 0:
            i0, i1 = 0, 1
        elif i == len(a)-1:
            i0, i1 = len(a)-2, len(a)-1
        else:
            i0, i1 = i-1, i+1
        f[i] = (np.log(D[i1]) - np.log(D[i0])) / (np.log(a[i1]) - np.log(a[i0]))
    return f

# ---------- Optional viscous correction (via tools/visc_fd_hook.py) ----------
_HAS_VISC = False
try:
    from tools.visc_fd_hook import apply_visc_fd_if_available  # uses env: DM_GRID, DM_KEFF
    _HAS_VISC = True
except Exception:
    _HAS_VISC = False

# ---------- IO helpers ----------
def read_fs8_table(path: str) -> pd.DataFrame:
    """
    Accepts many common fs8 formats. Tries to find z, fs8, err columns.
    If no error column is found, fills fs8_err with NaN (handled downstream).
    """
    df = pd.read_csv(path)
    # normalize headers
    cols = {c.lower().strip(): c for c in df.columns}
    z_keys   = [k for k in cols if k in ("z","z_eff","zbin","zbin_center","redshift")]
    fs8_keys = [k for k in cols if k in ("fs8","f_sigma8","f*sigma8","fσ8","f_sig8")]
    err_keys = [k for k in cols if k in ("fs8_err","f_sigma8_err","err","sigma","sigma_f_sigma8","fσ8_err")]

    if not z_keys or not fs8_keys:
        raise ValueError(f"Could not find z/fs8 columns in {path}. Columns: {list(df.columns)}")

    z_col   = cols[z_keys[0]]
    fs8_col = cols[fs8_keys[0]]
    if err_keys:
        err_col = cols[err_keys[0]]
        df = df[[z_col, fs8_col, err_col]].rename(columns={z_col:"z", fs8_col:"fs8", err_col:"fs8_err"})
    else:
        df = df[[z_col, fs8_col]].rename(columns={z_col:"z", fs8_col:"fs8"})
        df["fs8_err"] = np.nan

    # ensure numeric
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["fs8"] = pd.to_numeric(df["fs8"], errors="coerce")
    df["fs8_err"] = pd.to_numeric(df["fs8_err"], errors="coerce")
    df = df.dropna(subset=["z","fs8"]).reset_index(drop=True)
    return df.sort_values("z").reset_index(drop=True)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="fσ8 pipeline with optional viscous-DM correction.")
    ap.add_argument("--fs8-csv", type=str, default="fs8_measurements.csv", help="Input fσ8 table (CSV).")
    ap.add_argument("--Om0", type=float, default=0.3, help="Ω_m,0 (flat LCDM).")
    ap.add_argument("--h",   type=float, default=0.7, help="little h (H0/100).")
    ap.add_argument("--sigma8-0", type=float, default=0.80, help="σ8 today (normalization).")
    ap.add_argument("--a-min", type=float, default=1e-3, help="start a for growth solver.")
    ap.add_argument("--n-a",   type=int,   default=1200, help="number of a steps.")
    ap.add_argument("--out-root", type=str, default="outputs/fs8_visc", help="output root path (no extension).")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG plot output")
    args = ap.parse_args()

    Ode0 = 1.0 - args.Om0
    H0_per_Gyr = H0_to_per_Gyr(100.0 * args.h)

    # load data
    df = read_fs8_table(args.fs8_csv)

    # background growth
    a_grid = np.linspace(args.a_min, 1.0, args.n_a)
    D = solve_growth_LCDM(a_grid, H0_per_Gyr, args.Om0, Ode0)
    f = growth_rate_f(D, a_grid)

    # Evaluate model
    out_rows: List[dict] = []
    chi2 = 0.0
    ndata = 0

    for _, row in df.iterrows():
        z = float(row["z"])
        a = 1.0/(1.0+z)
        f_th = float(np.interp(a, a_grid, f))
        D_th = float(np.interp(a, a_grid, D))
        fs8_th = f_th * (args.sigma8_0 * D_th)

        # optional viscous correction
        if _HAS_VISC:
            fs8_th = apply_visc_fd_if_available(z, fs8_th)

        fs8_obs = float(row["fs8"])
        fs8_err = float(row["fs8_err"]) if pd.notna(row["fs8_err"]) else np.nan

        resid = fs8_obs - fs8_th
        # χ² only when error is finite & positive
        if np.isfinite(fs8_err) and fs8_err > 0:
            chi2 += (resid*resid) / (fs8_err*fs8_err)
            ndata += 1

        out_rows.append({
            "z": z, "a": a,
            "fs8_obs": fs8_obs, "fs8_err": fs8_err,
            "fs8_th": fs8_th, "residual": resid
        })

    # outputs
    out_root = args.out_root
    out_dir = os.path.dirname(out_root)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    res_csv = out_root + "_results.csv"
    pd.DataFrame(out_rows).to_csv(res_csv, index=False)

    # plotting
    fig_png = None
    if not args.no_plot:
        plt.figure()

        # Split data: with vs without finite error bars
        mask_err = np.isfinite(df["fs8_err"].values) & (df["fs8_err"].values > 0)
        df_err = df[mask_err]
        df_noerr = df[~mask_err]

        if len(df_err) > 0:
            plt.errorbar(df_err["z"], df_err["fs8"],
                         yerr=df_err["fs8_err"], fmt="o", capsize=3, label="data (err)")
        if len(df_noerr) > 0:
            plt.plot(df_noerr["z"], df_noerr["fs8"], "o", label="data (no err)")

        # theory curve
        z_plot = np.linspace(0, max(1.5, float(df["z"].max())*1.1), 200)
        a_plot = 1.0/(1.0+z_plot)
        f_plot = np.interp(a_plot, a_grid, f)
        D_plot = np.interp(a_plot, a_grid, D)
        fs8_plot = f_plot * (args.sigma8_0 * D_plot)

        if _HAS_VISC:
            try:
                from tools.visc_fd_hook import _ensure_loaded, visc_fd_ratio
                grid = _ensure_loaded()
                if grid is not None:
                    k_eff = float(os.environ.get("DM_KEFF", "0.20"))
                    ratios = np.array([visc_fd_ratio(z, k_eff, grid) for z in z_plot], float)
                    fs8_plot = fs8_plot * ratios
            except Exception:
                pass

        plt.plot(z_plot, fs8_plot, label="model", lw=2)
        plt.xlabel("z"); plt.ylabel("fσ8"); plt.legend()
        plt.title("fσ8 with optional viscous-DM correction")
        fig_png = out_root + "_plot.png"
        plt.tight_layout(); plt.savefig(fig_png, dpi=140); plt.close()

    # summary
    k_params = 3  # rough: {Om0, h, sigma8_0}
    aic = chi2 + 2*k_params if ndata > 0 else float("nan")
    bic = chi2 + k_params*math.log(ndata) if ndata > 0 else float("nan")

    meta = {
        "args": vars(args),
        "ndata_with_errors": int(ndata),
        "chi2": float(chi2) if ndata > 0 else float("nan"),
        "aic": float(aic),
        "bic": float(bic),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "viscous": bool(_HAS_VISC),
        "dm_grid": os.environ.get("DM_GRID"),
        "dm_keff": os.environ.get("DM_KEFF")
    }
    with open(out_root + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    print("  ", res_csv)
    if fig_png: print("  ", fig_png)
    print("  ", out_root + "_meta.json")
    if ndata > 0:
        print(f"  χ² = {chi2:.3f}  for N = {ndata}  (AIC ~ {aic:.2f}, BIC ~ {bic:.2f})")
    else:
        print("  Note: no finite error bars found; plotted points without y-errors and skipped χ².")

if __name__ == "__main__":
    main()
