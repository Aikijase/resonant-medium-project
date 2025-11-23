#!/usr/bin/env python3
import argparse, os, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- helpers ----------------
def H_of_z(z, H0, Om0, Ode_part, Or0=0.0):
    E2 = Om0*(1+z)**3 + Or0*(1+z)**4 + Ode_part
    return H0*np.sqrt(np.maximum(E2, 1e-30))

def rho_crit0_msun_mpc3(H0):
    G = 4.30091e-9  # Mpc*(km/s)^2/Msun
    return 3.0*(H0**2)/(8.0*np.pi*G)

def pick_col(tbl, contains_all=None, startswith=None, fallback_substr=None):
    cols = list(tbl.columns)
    low  = [c.lower() for c in cols]
    if contains_all:
        for c in cols:
            n=c.lower()
            if all(tok in n for tok in contains_all): return c
    if startswith:
        for c in cols:
            if c.lower().startswith(startswith.lower()): return c
    if fallback_substr:
        for c in cols:
            if fallback_substr.lower() in c.lower(): return c
    num = tbl.select_dtypes(include=[float,int]).columns
    return num[0] if len(num)>0 else cols[0]

def read_qc(qc_path):
    if not os.path.exists(qc_path): return None
    out = {}
    with open(qc_path) as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for k,v in rdr:
            try: out[k]=float(v)
            except: pass
    if "delta_rho_from_accretion_Msun_Mpc3" in out and "observed_delta_rhoBH_Msun_Mpc3" in out:
        return {
            "delta_rho_from_accretion": out["delta_rho_from_accretion_Msun_Mpc3"],
            "observed_delta_rhoBH": out["observed_delta_rhoBH_Msun_Mpc3"],
            "ratio_acc_to_obs": out["ratio_accretion_to_observed"]
        }
    return None

def norm(v):
    m = np.nanmax(np.abs(v))
    return v/m if m>0 else v

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rho-de', required=True)
    ap.add_argument('--rho-bh', required=True)
    ap.add_argument('--rho-dot', default='data/bh_accretion_from_lf.csv')
    ap.add_argument('--qc', default='logs/qc_rhoBH_vs_accretion.csv')
    ap.add_argument('--H0', type=float, default=70.0)
    ap.add_argument('--Om0', type=float, default=0.3)
    ap.add_argument('--Or0', type=float, default=0.0)
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    # --- Load DE ---
    de = pd.read_csv(args.rho_de)
    zcol_de = 'z' if 'z' in de.columns else pick_col(de, startswith='z')
    rho_de_col = pick_col(de, contains_all=['rho','de'])
    z_de = de[zcol_de].values
    rho_de_series = de[rho_de_col].values

    # --- Detect units ---
    rho_de_at0 = float(np.interp(0, z_de, rho_de_series))
    is_omega_like = rho_de_at0 < 2.0

    if is_omega_like:
        Ode0_target = 1 - args.Om0 - args.Or0
        scale = Ode0_target / max(rho_de_at0, 1e-30)
        Ode = np.interp(z_de, z_de, rho_de_series) * scale
        rho_de_phys = Ode * rho_crit0_msun_mpc3(args.H0)
    else:
        rho_de_phys = rho_de_series

    # --- Load DM if available ---
    rho_dm_col = None
    for c in de.columns:
        if 'rho_dm' in c.lower(): rho_dm_col=c
    if rho_dm_col:
        rho_dm_series = de[rho_dm_col].values
    else:
        rho_dm_series = np.zeros_like(rho_de_phys)

    # --- Common z grid ---
    z = np.linspace(0, z_de.max(), 200)
    rho_de_grid = np.interp(z, z_de, rho_de_phys)
    rho_dm_grid = np.interp(z, z_de, rho_dm_series)

    # --- Closure from QC ---
    qc = read_qc(args.qc)
    if not qc:
        raise SystemExit("Missing QC closure file. Run QC first.")
    delta_rho_acc = qc["delta_rho_from_accretion"]
    delta_rho_bh  = qc["observed_delta_rhoBH"]

    # --- Δρ ---
    delta_rho_de   = rho_de_grid[0] - rho_de_grid[-1]
    delta_rho_dark = (rho_de_grid + rho_dm_grid)[0] - (rho_de_grid + rho_dm_grid)[-1]
    eta_de         = delta_rho_de / max(delta_rho_acc,1e-30)
    eta_dark       = delta_rho_dark / max(delta_rho_acc,1e-30)

    # --- Plot shapes ---
    d_de_dz = np.gradient(rho_de_grid, z)
    plt.figure()
    plt.plot(z, norm(d_de_dz), label="d rho_de/dz (norm)")
    plt.plot(z, norm(np.gradient(rho_de_grid+rho_dm_grid,z)), label="d rho_dark/dz (norm)")
    plt.gca().invert_xaxis()
    plt.xlabel("z"); plt.ylabel("normalized slope")
    plt.legend()
    plt.savefig(args.out_prefix+".plot.png")

    # --- Report ---
    rep = {
        "closure": qc,
        "dark_energy_link": {
            "delta_rho_de": float(delta_rho_de),
            "eta_de": float(eta_de),
            "delta_rho_dark": float(delta_rho_dark),
            "eta_dark": float(eta_dark)
        }
    }
    with open(args.out_prefix+".report.json","w") as f: json.dump(rep,f,indent=2)
    print(json.dumps(rep,indent=2))
    print(f"Wrote {args.out_prefix+'.report.json'} and .plot.png")

if __name__ == "__main__":
    main()
