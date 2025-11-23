import os, json, math, argparse
import numpy as np, pandas as pd
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import tomllib
from models.gamma_profiles import PROFILES

def Ez(z, Om0=0.3, Ol0=0.7):
    return np.sqrt(Om0*(1+z)**3 + Ol0)

def Omega_m_z(z, Om0=0.3):
    z = np.asarray(z, dtype=float)
    Ez2 = Om0*(1+z)**3 + (1-Om0)
    return (Om0*(1+z)**3)/Ez2

def D_from_f(z, f_of_z):
    """Integrate D ∝ exp(∫ f dln a), normalized D(0)=1."""
    z = np.asarray(z); a = 1.0/(1.0+z)
    idx = np.argsort(z); z = z[idx]; a = a[idx]
    f = f_of_z[idx]
    dlnA = np.diff(np.log(a))
    f_mid = 0.5*(f[1:]+f[:-1])
    lnD = np.zeros_like(a)
    lnD[1:] = np.cumsum(f_mid * dlnA)
    D = np.exp(lnD)
    inv = np.empty_like(idx); inv[idx] = np.arange(len(idx))
    return D[inv]

def compute_fs8_gammaevo(cfg):
    Om0 = cfg["cosmo"]["Omega_m0"]
    sig8_0 = cfg["cosmo"]["sigma8_0"]
    z = np.linspace(cfg["zgrid"]["zmin"], cfg["zgrid"]["zmax"], cfg["zgrid"]["npts"])
    Omz = Omega_m_z(z, Om0)
    prof = PROFILES["smooth"]

    G0 = np.linspace(cfg["grid"]["gamma0_min"], cfg["grid"]["gamma0_max"], cfg["grid"]["gamma0_n"])
    G1 = np.linspace(cfg["grid"]["g1_min"], cfg["grid"]["g1_max"], cfg["grid"]["g1_n"])
    rows = []

    Dobs = pd.read_csv(cfg["data"]["fs8_csv"])
    z_d, d, s = Dobs["z"].values, Dobs["fs8_obs"].values, Dobs["sigma"].values
    w = 1.0/(s*s)

    ref = None
    if os.path.isfile(cfg["scoring"]["lcdm_ref_json"]):
        try:
            ref = json.load(open(cfg["scoring"]["lcdm_ref_json"]))
        except Exception:
            pass

    for g0 in G0:
        for g1 in G1:
            gZ = prof(z, gamma0=g0, g1=g1)
            fz = np.power(Omz, gZ)
            Dz = D_from_f(z, fz)
            fs8 = sig8_0 * fz * Dz

            y = np.interp(z_d, z, fs8)
            alpha, k = 1.0, 0
            if cfg["scoring"]["fit_s8"]:
                num = float(np.sum(w * d * y))
                den = float(np.sum(w * y * y) + 1e-12)
                alpha = num/den; k = 1
            yfit = alpha * y
            chi2 = float(np.sum(((yfit - d)/s)**2))
            n = int(len(d))
            AIC = chi2 + 2*k
            BIC = chi2 + k*math.log(max(n,1))

            dA = dB = WWI = None
            if ref and "AIC" in ref and "BIC" in ref:
                dA = AIC - ref["AIC"]["value"]
                dB = BIC - ref["BIC"]["value"]
                WWI = 100*min(1.0, max(0.0, -dA/10.0), max(0.0, -dB/10.0))

            rows.append({
                "gamma0": g0, "g1": g1, "alpha": alpha,
                "chi2": chi2, "AIC": AIC, "BIC": BIC,
                "dAIC": dA, "dBIC": dB, "WWI": WWI
            })

    df = pd.DataFrame(rows)
    os.makedirs("outputs/phase3", exist_ok=True)
    df.to_csv("outputs/phase3/fs8_gammaevo_grid.csv", index=False)
    best = df.sort_values("AIC").iloc[0].to_dict()
    json.dump({"best": best}, open("outputs/phase3/fs8_gammaevo_best.json", "w"), indent=2)
    print("Wrote outputs/phase3/fs8_gammaevo_grid.csv and ..._best.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase3.toml")
    args = ap.parse_args()
    cfg = tomllib.load(open(args.cfg, "rb"))
    compute_fs8_gammaevo(cfg)
