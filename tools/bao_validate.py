#!/usr/bin/env python3
"""
bao_validate.py — Anisotropic BAO validation for Thrace.

Inputs:
  • Background: LCDM (Ωm0) or TABLE (a,E2) or (a,H)
  • H0 (km/s/Mpc), r_drag (Mpc)
  • Observations CSV with columns (any subset):
      z,
      DM_over_rd, err_DM_over_rd | sigma_DM_over_rd,
      DH_over_rd, err_DH_over_rd | sigma_DH_over_rd,
      DV_over_rd, err_DV_over_rd | sigma_DV_over_rd
    (Diagonal errors assumed; covariance not supported in this simple pass.)

Outputs (for --out-prefix PREFIX):
  PREFIX_bao_predictions.csv      # z, DM/rd, DH/rd, DV/rd
  PREFIX_bao_residuals.csv        # per-point residuals & pulls
  PREFIX_bao_scorecard.txt        # χ² summary
  PREFIX_bao_overlay_DM.png       # DM/rd overlay
  PREFIX_bao_overlay_DH.png       # DH/rd overlay (if available)
  PREFIX_bao_overlay_DV.png       # DV/rd overlay (if available)
  PREFIX_bao_compare.txt          # vs LCDM baseline
  RESULT: PASS/FAIL line printed to stdout

Example (TABLE mode):
  python tools/bao_validate.py \
    --mode table --table data/bg.csv \
    --omega-m0 0.30 --H0 70 --rdrag 147.1 \
    --obs data/bao_aniso.csv \
    --zmax 2.0 --N 2000 \
    --out-prefix outputs/thrace_best/bao_thrace

Example (LCDM baseline only):
  python tools/bao_validate.py \
    --mode lcdm --omega-m0 0.30 --H0 70 --rdrag 147.1 \
    --obs data/bao_aniso.csv \
    --out-prefix outputs/thrace_best/bao_lcdm
"""

import argparse, csv, math, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

C_KMS = 299792.458  # km/s

# ---------- IO helpers ----------
def ensure_dir_for(path: str):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)

def read_csv_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r); cols = r.fieldnames
    return rows, cols

def write_csv(path, header, rows_iter):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for row in rows_iter:
            w.writerow(row)

def sniff(cols, *cands):
    for c in cands:
        if c in cols: return c
    return None

# ---------- Background E2(a) ----------
def build_E2_from_table(path):
    rows, cols = read_csv_rows(path)
    ca = "a" if "a" in cols else None
    cz = "z" if "z" in cols else None
    cE2 = "E2" if "E2" in cols else None
    cH  = "H"  if "H"  in cols else None
    if not (ca or cz):
        raise ValueError("table must have 'a' or 'z' column")
    if not (cE2 or cH):
        raise ValueError("table must include 'E2' or 'H'")

    A, V = [], []
    for r in rows:
        a = float(r[ca]) if ca else 1.0/(1.0 + float(r[cz]))
        v = float(r[cE2]) if cE2 else float(r[cH])  # normalize later if H
        A.append(a); V.append(v)
    A = np.asarray(A); V = np.asarray(V)
    order = np.argsort(A); A = A[order]; V = V[order]

    if cH and not cE2:
        # normalize H to H(1) => E2 = (H/H(1))^2
        idx = np.argmin(np.abs(A-1.0))
        H1 = V[idx]
        V = (V/H1)**2

    # enforce E2(1)=1
    idx = np.argmin(np.abs(A-1.0))
    if V[idx] != 0: V = V / V[idx]

    def E2f(aq): return np.interp(aq, A, V)
    return E2f

def E2_LCDM(om0):
    ode0 = 1.0 - om0
    return lambda a: om0 * a**(-3) + ode0

# ---------- Distances ----------
def H_of_z(z, H0, E2f):
    a = 1.0/(1.0+z)
    return H0 * math.sqrt(max(E2f(a), 1e-300))  # km/s/Mpc

def DM_of_z(z, H0, E2f, nsteps=4096):
    # Flat geometry: D_M = comoving radial distance D_C
    # D_C(z) = c ∫0^z dz'/H(z')
    zs = np.linspace(0.0, z, max(4, int(nsteps*z+4)))
    Hz = np.array([H_of_z(zi, H0, E2f) for zi in zs])
    integ = np.trapz(1.0/Hz, zs)  # (Mpc s / km) but c fixes units
    return C_KMS * integ  # Mpc

def DV_of_z(z, H0, E2f):
    # DV(z) = [ D_M(z)^2 * c z / H(z) ]^{1/3}
    DM = DM_of_z(z, H0, E2f)
    Hz = H_of_z(z, H0, E2f)
    return (DM*DM * C_KMS * z / Hz) ** (1.0/3.0)

# ---------- Scoring & plots ----------
def score_bao(obs_csv, H0, rdrag, E2f, out_prefix):
    rows, cols = read_csv_rows(obs_csv)
    zc  = sniff(cols, "z","z_eff","zeff","zmid")
    # metrics present?
    dm_c = sniff(cols, "DM_over_rd","DM/rd","D_M_over_rdrag","D_M_over_rd","DMrd")
    dh_c = sniff(cols, "DH_over_rd","DH/rd","D_H_over_rdrag","D_H_over_rd","DHrd")
    dv_c = sniff(cols, "DV_over_rd","DV/rd","D_V_over_rdrag","D_V_over_rd","DVrd")

    dm_e = sniff(cols, "err_DM_over_rd","sigma_DM_over_rd","eDM","sd_DM","DM_over_rd_err")
    dh_e = sniff(cols, "err_DH_over_rd","sigma_DH_over_rd","eDH","sd_DH","DH_over_rd_err")
    dv_e = sniff(cols, "err_DV_over_rd","sigma_DV_over_rd","eDV","sd_DV","DV_over_rd_err")

    if not zc or not (dm_c or dh_c or dv_c):
        raise ValueError(f"obs needs z and at least one of DM_over_rd, DH_over_rd, DV_over_rd. cols={cols}")

    out_rows = []
    chi2 = 0.0
    npts = 0

    pred_csv = f"{out_prefix}_bao_predictions.csv"
    ensure_dir_for(pred_csv)
    pred_rows = []

    # Pre-compute predictions at observed z
    for o in rows:
        try:
            z = float(o[zc])
        except Exception:
            continue
        DM_rd = DM_of_z(z, H0, E2f)/rdrag
        DH_rd = (C_KMS/H_of_z(z, H0, E2f))/rdrag
        DV_rd = DV_of_z(z, H0, E2f)/rdrag
        pred_rows.append([z, DM_rd, DH_rd, DV_rd])

        # accumulate chi2 over available metrics
        if dm_c and dm_e and o.get(dm_c, "") not in ("", None) and o.get(dm_e, "") not in ("", None):
            obs = float(o[dm_c]); err = float(o[dm_e]); mod = DM_rd
            res = mod - obs; pull = res/err
            chi2 += pull*pull; npts += 1
            out_rows.append(["DM/rd", z, obs, err, mod, res, pull])

        if dh_c and dh_e and o.get(dh_c, "") not in ("", None) and o.get(dh_e, "") not in ("", None):
            obs = float(o[dh_c]); err = float(o[dh_e]); mod = DH_rd
            res = mod - obs; pull = res/err
            chi2 += pull*pull; npts += 1
            out_rows.append(["DH/rd", z, obs, err, mod, res, pull])

        if dv_c and dv_e and o.get(dv_c, "") not in ("", None) and o.get(dv_e, "") not in ("", None):
            obs = float(o[dv_c]); err = float(o[dv_e]); mod = DV_rd
            res = mod - obs; pull = res/err
            chi2 += pull*pull; npts += 1
            out_rows.append(["DV/rd", z, obs, err, mod, res, pull])

    # write predictions & residuals & score
    write_csv(pred_csv, ["z","DM_over_rd_pred","DH_over_rd_pred","DV_over_rd_pred"], pred_rows)
    write_csv(f"{out_prefix}_bao_residuals.csv",
              ["metric","z","obs","sigma","model","resid","pull"], out_rows)
    with open(f"{out_prefix}_bao_scorecard.txt","w") as f:
        f.write("BAO Scorecard (anisotropic)\n")
        f.write(f" points: {npts}\n")
        f.write(f" chi2  : {chi2:.3f}\n")
        ratio = chi2/max(npts,1)
        f.write(f" chi2/dof (diag) : {ratio:.3f}\n")
    return npts, chi2, (chi2/max(npts,1) if npts else float("inf"))

def overlay_plot(z_curve, y_curve, z_obs, y_obs, y_err, title, ylabel, out_png):
    import numpy as np
    plt.figure()
    if len(z_curve) and len(y_curve):
        plt.plot(z_curve, y_curve, label="Model")
    if z_obs is not None and y_obs is not None and len(z_obs) and len(y_obs):
        if y_err is not None and len(y_err) == len(y_obs):
            plt.errorbar(z_obs, y_obs, yerr=y_err, fmt="o", capsize=3, label="Obs")
        else:
            plt.plot(z_obs, y_obs, "o", label="Obs")
    plt.xlabel("z"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.tight_layout()
    ensure_dir_for(out_png)
    plt.savefig(out_png, dpi=150)

def make_overlays(obs_csv, H0, rdrag, E2f, out_prefix):
    rows, cols = read_csv_rows(obs_csv)
    zc  = sniff(cols, "z","z_eff","zeff","zmid")
    dm_c = sniff(cols, "DM_over_rd","DM/rd","D_M_over_rdrag","D_M_over_rd","DMrd")
    dh_c = sniff(cols, "DH_over_rd","DH/rd","D_H_over_rdrag","D_H_over_rd","DHrd")
    dv_c = sniff(cols, "DV_over_rd","DV/rd","D_V_over_rdrag","D_V_over_rd","DVrd")
    dm_e = sniff(cols, "err_DM_over_rd","sigma_DM_over_rd","eDM","sd_DM","DM_over_rd_err")
    dh_e = sniff(cols, "err_DH_over_rd","sigma_DH_over_rd","eDH","sd_DH","DH_over_rd_err")
    dv_e = sniff(cols, "err_DV_over_rd","sigma_DV_over_rd","eDV","sd_DV","DV_over_rd_err")

    # keep only rows with a usable z
    valid = []
    for r in rows:
        try:
            _ = float(r.get(zc, ""))
            valid.append(r)
        except Exception:
            continue

    # dense model curve up to max z in data
    if valid:
        zmax = max(0.01, max(float(r[zc]) for r in valid))
    else:
        zmax = 0.01
    zs = np.linspace(0.01, zmax, 800)
    DM_rd_curve = np.array([DM_of_z(z, H0, E2f)/rdrag for z in zs])
    DH_rd_curve = np.array([(C_KMS/H_of_z(z, H0, E2f))/rdrag for z in zs])
    DV_rd_curve = np.array([DV_of_z(z, H0, E2f)/rdrag for z in zs])

    def build_obs_arrays(yc, ec):
        z_list, y_list, e_list = [], [], []
        if not yc:
            return None, None, None
        for r in valid:
            try:
                zf = float(r.get(zc, ""))
                yf = float(r.get(yc, ""))
            except Exception:
                continue
            if ec:
                try:
                    ef = float(r.get(ec, ""))
                except Exception:
                    continue
                if not np.isfinite(ef) or ef <= 0:
                    continue
                z_list.append(zf); y_list.append(yf); e_list.append(ef)
            else:
                z_list.append(zf); y_list.append(yf)
        z_arr = np.array(z_list)
        y_arr = np.array(y_list)
        e_arr = (np.array(e_list) if ec else None)
        return z_arr, y_arr, e_arr

    # DM overlay
    z_obs, y_obs, e_obs = build_obs_arrays(dm_c, dm_e)
    if z_obs is not None and y_obs is not None and len(z_obs) and len(y_obs):
        overlay_plot(zs, DM_rd_curve, z_obs, y_obs, e_obs,
                     "BAO: D_M / r_d", "D_M / r_d", f"{out_prefix}_bao_overlay_DM.png")

    # DH overlay
    z_obs, y_obs, e_obs = build_obs_arrays(dh_c, dh_e)
    if z_obs is not None and y_obs is not None and len(z_obs) and len(y_obs):
        overlay_plot(zs, DH_rd_curve, z_obs, y_obs, e_obs,
                     "BAO: D_H / r_d", "D_H / r_d", f"{out_prefix}_bao_overlay_DH.png")

    # DV overlay (isotropic)
    z_obs, y_obs, e_obs = build_obs_arrays(dv_c, dv_e)
    if z_obs is not None and y_obs is not None and len(z_obs) and len(y_obs):
        overlay_plot(zs, DV_rd_curve, z_obs, y_obs, e_obs,
                     "BAO: D_V / r_d", "D_V / r_d", f"{out_prefix}_bao_overlay_DV.png")

    # Plot DM
    if dm_c and dm_e:
        overlay_plot(zs, DM_rd_curve, z_obs, z_obs, e_obs,
                     "BAO: D_M / r_d", "D_M / r_d", f"{out_prefix}_bao_overlay_DM.png")

    # Plot DH
    if dh_c and dh_e:
        overlay_plot(zs, DH_rd_curve, z_obs, z_obs, e_obs,
                     "BAO: D_H / r_d", "D_H / r_d", f"{out_prefix}_bao_overlay_DH.png")

    # Plot DV (optional)
    if dv_c and dv_e:
        overlay_plot(zs, DV_rd_curve, z_obs, z_obs, e_obs,
                     "BAO: D_V / r_d", "D_V / r_d", f"{out_prefix}_bao_overlay_DV.png")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lcdm","table"], required=True)
    ap.add_argument("--omega-m0", type=float, required=True)
    ap.add_argument("--H0", type=float, required=True, help="H0 in km/s/Mpc")
    ap.add_argument("--rdrag", type=float, required=True, help="Sound horizon at drag (Mpc)")
    ap.add_argument("--table", type=str, default=None, help="If --mode table: CSV with a,E2 or a,H")
    ap.add_argument("--obs", type=str, required=True, help="BAO observations CSV")
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--out-prefix", type=str, default="outputs/thrace_best/bao")
    args = ap.parse_args()

    # Build E2(a)
    if args.mode == "lcdm":
        E2f_model = E2_LCDM(args.omega_m0)
    else:
        if not args.table:
            print("ERROR: --table is required in table mode", file=sys.stderr); sys.exit(2)
        E2f_model = build_E2_from_table(args.table)

    # Also LCDM baseline
    E2f_lcdm = E2_LCDM(args.omega_m0)

    # Score model
    n_m, chi2_m, r_m = score_bao(args.obs, args.H0, args.rdrag, E2f_model, args.out_prefix)
    make_overlays(args.obs, args.H0, args.rdrag, E2f_model, args.out_prefix)

    # Score LCDM baseline (same Ωm0, H0, r_d so scale is fair)
    base_prefix = str(Path(args.out_prefix).with_name("bao_lcdm"))
    n_l, chi2_l, r_l = score_bao(args.obs, args.H0, args.rdrag, E2f_lcdm, base_prefix)
    make_overlays(args.obs, args.H0, args.rdrag, E2f_lcdm, base_prefix)

    dchi2 = chi2_m - chi2_l
    compare = (
        "BAO Comparison (Model vs ΛCDM)\n"
        f"  points: {n_m}\n"
        f"  Model : chi2={chi2_m:.3f}, chi2/pt={r_m:.3f}\n"
        f"  ΛCDM  : chi2={chi2_l:.3f}, chi2/pt={r_l:.3f}\n"
        f"  Δchi2 : {dchi2:+.3f}  (negative = Model better)\n"
    )
    with open(f"{args.out_prefix}_bao_compare.txt","w") as f:
        f.write(compare)

    print(f"RESULT: PASS (BAO diag χ²/pt={r_m:.3f}; Δχ²={dchi2:+.3f} vs ΛCDM; n={n_m})")
    print(f"Wrote predictions/residuals/scorecard/overlays under prefix {args.out_prefix}")
    print(f"Wrote LCDM baseline under prefix {base_prefix}")

if __name__ == "__main__":
    main()
