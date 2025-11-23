#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def pick_col_like(df, names):
    names = [n.lower().replace("_","") for n in names]
    for c in df.columns:
        n = c.lower().replace("_","")
        if any(n == t or t in n for t in names):
            return c
    raise KeyError(f"No column like {names} in {list(df.columns)}")

def load_rho_de(path):
    df = pd.read_csv(path)
    zc = pick_col_like(df, ["z","redshift"])
    dc = None
    for c in df.columns:
        n=c.lower()
        if ("rho_de" in n) or (("rho" in n) and ("de" in n)):
            dc=c; break
    if dc is None: dc = pick_col_like(df, ["rhode","de","darkenergy"])
    out = df[[zc,dc]].rename(columns={zc:"z",dc:"rho_de"}).dropna()
    return out.sort_values("z")

def load_fs8(path):
    df = pd.read_csv(path)
    zc = pick_col_like(df, ["z","redshift"])
    fc = pick_col_like(df, ["fs8","f_sigma8"])
    try:
        sc = pick_col_like(df, ["sigma","sigma_fs8","err","error"])
    except KeyError:
        raise SystemExit(f"No sigma column in {path}")
    out = df[[zc,fc,sc]].rename(columns={zc:"z",fc:"fs8",sc:"sigma"}).dropna()
    # if percent-style, convert to fraction
    if out["fs8"].median() > 1.0:
        out["fs8"] /= 100.0
        if out["sigma"].median() > 1.0:
            out["sigma"] /= 100.0
    # guardrails
    out = out[(out["sigma"]>1e-3) & (out["fs8"]>0) & (out["fs8"]<1.5)]
    return out.sort_values("z")

# ---------- growth model (stable: early -> today) ----------
def model_fs8(rho_de_tbl, H0=70.0, Om0=0.3, Or0=0.0, sigma8_0=0.8):
    t = rho_de_tbl[["z","rho_de"]].dropna().sort_values("z").copy()
    # ensure z=0 present by interpolation (not copying first value)
    if t["z"].min() > 0.0:
        rh0 = float(np.interp(0.0, t["z"].values, t["rho_de"].values))
        t = pd.concat([pd.DataFrame({"z":[0.0],"rho_de":[rh0]}), t], ignore_index=True)
    # normalize to Ω_de(0)=1-Ω_m-Ω_r (assumes input is proportional to Ω_de)
    Ode0 = max(1.0 - Om0 - Or0, 1e-6)
    Ode0_raw = float(np.interp(0.0, t["z"].values, t["rho_de"].values))
    scale = Ode0 / max(Ode0_raw, 1e-30)

    zmax = float(t["z"].max())
    a_min = 1.0/(1.0+zmax)
    a = np.linspace(a_min, 1.0, 1200)          # early -> today
    z = 1.0/a - 1.0
    Ode = np.interp(z, t["z"].values, t["rho_de"].values) * scale
    E2  = Om0*(1+z)**3 + Or0*(1+z)**4 + Ode
    E   = np.sqrt(np.maximum(E2,1e-38))

    # d ln H / d ln a
    dOdz = np.gradient(Ode, z)
    dE2dz = 3*Om0*(1+z)**2 + 4*Or0*(1+z)**3 + dOdz
    dlnHdlna = 0.5*(dE2dz/E2)*(-(1+z))
    Omz = Om0*(1+z)**3 / E2

    # integrate D'' + [2 + dlnH/dlna] D' - 1.5*Om(a) D = 0 in ln a
    lna = np.log(a)
    D  = np.zeros_like(a); Dp = np.zeros_like(a)
    # initial conditions at early times: D ~ a, D' ~ D
    D[0]  = a[0]
    Dp[0] = a[0]
    for i in range(1,len(a)):
        dl = lna[i] - lna[i-1]
        c1 = 2.0 + dlnHdlna[i]
        c2 = 1.5 * Omz[i]
        # RK2
        k1_D  = Dp[i-1]
        k1_Dp = -c1*Dp[i-1] + c2*D[i-1]
        D_mid  = D[i-1]  + 0.5*dl*k1_D
        Dp_mid = Dp[i-1] + 0.5*dl*k1_Dp
        k2_D  = Dp_mid
        k2_Dp = -c1*Dp_mid + c2*D_mid
        D[i]  = D[i-1]  + dl*k2_D
        Dp[i] = Dp[i-1] + dl*k2_Dp

    # normalize so D(a=1)=1
    D  /= D[-1]
    Dp /= D[-1]
    f = Dp/np.maximum(D,1e-30)
    fs8 = sigma8_0 * D * f
    return pd.DataFrame({"z":z, "fs8":fs8}).sort_values("z")

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rho-de", required=True)
    ap.add_argument("--fs8-data", default=None)
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--Om0", type=float, default=0.3)
    ap.add_argument("--Or0", type=float, default=0.0)
    ap.add_argument("--sigma8_0", type=float, default=0.80)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--out-prefix", required=True)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    rho_de = load_rho_de(args.rho_de)
    model  = model_fs8(rho_de, H0=args.H0, Om0=args.Om0, Or0=args.Or0, sigma8_0=args.sigma8_0)
    model.to_csv(args.out_prefix+".model_fs8.csv", index=False)

    n=0; chi2=np.nan; wwi=np.nan; mm=None
    if args.fs8_data and os.path.exists(args.fs8_data):
        obs = load_fs8(args.fs8_data)
        mm = pd.merge_asof(
            obs.sort_values("z"),
            model.sort_values("z"),
            on="z", direction="nearest",
            tolerance=args.tolerance
        ).dropna()
        if len(mm)>0:
            resid = (mm["fs8_x"] - mm["fs8_y"]) / mm["sigma"]
            chi2  = float(np.sum(resid**2))
            n     = int(len(mm))
            wwi   = float(100.0*np.exp(-0.5*chi2/max(1,n)))
            mm.rename(columns={"fs8_x":"fs8_obs","fs8_y":"fs8_model"}).to_csv(args.out_prefix+".compare.csv", index=False)

    rep = {"inputs":{"rho_de":args.rho_de,"fs8_data":args.fs8_data,"H0":args.H0,"Om0":args.Om0,"sigma8_0":args.sigma8_0,"tolerance":args.tolerance},
           "n_points":n,"chi2":chi2,"WWI_like":wwi}
    with open(args.out_prefix+".report.json","w") as f: json.dump(rep,f,indent=2)

    plt.figure()
    plt.plot(model["z"], model["fs8"], label="model")
    if mm is not None and len(mm)>0:
        plt.errorbar(mm["z"], mm["fs8_x"], yerr=mm["sigma"], fmt="o", label="obs")
    plt.gca().invert_xaxis(); plt.xlabel("z"); plt.ylabel("fσ8"); plt.legend()
    plt.savefig(args.out_prefix+".plot.png", dpi=160, bbox_inches="tight")

    if n>0:
        print(f"RESULT: n={n} chi2={chi2:.2f} WWI~{wwi:.1f}")
    else:
        print("RESULT: n=0 (no matched points)")

if __name__=="__main__":
    main()
