import os, sys, json, math, glob
import numpy as np, pandas as pd

# project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# growth/isw proxies
from isw import phi_proxy as phi_proxy_res

# vector LCDM proxy (shape baseline)
def H_lcdm(z, H0=70.0, Om=0.3, Ol=0.7):
    z = np.asarray(z, dtype=float)
    return H0 * np.sqrt(Om*(1+z)**3 + Ol)

def phi_proxy_lcdm(z):
    z = np.asarray(z, dtype=float)
    Hz = H_lcdm(z)
    Hz = np.clip(Hz, 1e-9, None)
    return (1.0/(Hz**2)) * np.exp(-z)

# load base params + (f,γ) grid from your basins CSV (keeps grids aligned)
import tomllib
with open("configs/resonant.toml","rb") as fh:
    BASE = tomllib.load(fh)["params"]

B = pd.read_csv("outputs/phase2/basin_grid.csv")
fvals = np.sort(B["f"].unique())
gvals = np.sort(B["gamma"].unique())

# amplitude-format ISW rows (WISE + RACS)
rows = []
required = {"survey","ell_min","ell_max","A_ISW","sigma_A","z_eff"}
for path in sorted(glob.glob("data/isw/*.csv")):
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    if not required.issubset(set(df.columns)):
        continue
    for _,r in df.iterrows():
        rows.append({
            "survey": str(r["survey"]),
            "A_ISW": float(r["A_ISW"]),
            "sigma_A": float(r["sigma_A"]),
            "z_eff": float(r["z_eff"]),
        })
if not rows:
    raise SystemExit("No ISW amplitude rows found in data/isw/*.csv")
ISW = pd.DataFrame(rows)

# half-widths (same as scorer; tweak if you like)
SURVEY_HALFWIDTH = {
    "WISE_GAL":0.20, "WISE_AGN":0.35, "RACS_SKADS":0.30, "RACS_BACCUS":0.30
}
DEFAULT_HALFWIDTH = 0.25

def predict_A_model(P, zc, halfw, nz=128):
    z1, z2 = max(0.0, zc-halfw), zc+halfw
    if not np.isfinite(z1) or not np.isfinite(z2) or z2<=z1: return np.nan
    z = np.linspace(z1, z2, nz)
    res = np.asarray(phi_proxy_res(z, P), dtype=float)
    lcd = phi_proxy_lcdm(z).astype(float)
    if (not np.all(np.isfinite(res))) or (not np.all(np.isfinite(lcd))): return np.nan
    if np.any(lcd<=0): return np.nan

    # winsorize & normalize (shape-only)
    def wins(x):
        lo,hi = np.percentile(x,5), np.percentile(x,95)
        return np.clip(x, max(lo,1e-12), hi)
    rw, lw = wins(res), wins(lcd)
    mr, ml = np.median(rw), np.median(lw)
    if mr<=0 or ml<=0 or not np.isfinite(mr) or not np.isfinite(ml): return np.nan
    r = (rw/mr) / (lw/ml)
    r = np.clip(r, 0.2, 2.0)
    return float(np.mean(r))

# ISW acceptance threshold
# ok if chi2/n <= 2.0 (≈ within ~√2 sigma per datum); tune as desired
ISW_CHI2_PER_DOF_MAX = 2.0

records=[]
for f in fvals:
    for g in gvals:
        P = dict(BASE); P["f"]=float(f); P["gamma"]=float(g)
        chis=0.0; n=0
        for _,r in ISW.iterrows():
            zc = float(r["z_eff"])
            hw = SURVEY_HALFWIDTH.get(r["survey"], DEFAULT_HALFWIDTH)
            A_mod = predict_A_model(P, zc, hw)
            if not np.isfinite(A_mod): 
                continue
            sigma = max(float(r["sigma_A"]), 1e-9)
            chis += ((A_mod - float(r["A_ISW"])) / sigma)**2
            n += 1
        chi2 = float(chis)
        ok = (n>0) and (chi2/float(n) <= ISW_CHI2_PER_DOF_MAX)
        records.append({"f":f,"gamma":g,"chi2_isw":chi2,"n_isw":n,"isw_ok": ok})

os.makedirs("outputs/phase2", exist_ok=True)
outcsv = "outputs/phase2/isw_ok_grid.csv"
pd.DataFrame(records).to_csv(outcsv, index=False)
print("Wrote", outcsv)
