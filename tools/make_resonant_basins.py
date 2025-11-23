import os, sys, math, json
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from growth import solve_growth, H_of_z

# ---- Grid (match your stability sweep) ----
F_GRID   = np.linspace(2.40, 2.85, 19)
GAM_GRID = np.linspace(1.40, 1.65, 11)
ZMAX, NPTS, SIGMA8_0 = 2.0, 200, 0.8
PERT_EPS = 0.01  # stress jitter

# Load base params
import tomllib
with open("configs/resonant.toml","rb") as fh:
    BASE = tomllib.load(fh)["params"]

# Data + LCDM reference for deltas
D = pd.read_csv("data/growth/fs8_catalog.csv")
ref = None
ref_path = "outputs/phase2/fs8_eval_lcdm.json"
if os.path.exists(ref_path):
    ref = json.load(open(ref_path))

def aic_bic(chi2, k, n):
    AIC = chi2 + 2*k
    BIC = chi2 + k*math.log(max(n,1))
    return AIC, BIC

def wwi(dA, dB):
    if dA is None or dB is None: return None
    return 100*min(1.0, max(0.0, -dA/10.0), max(0.0, -dB/10.0))

def metric_stability(p):
    zs = np.linspace(0, ZMAX, 100)
    Hs = np.array([H_of_z(z, p) for z in zs])
    h_ok = np.all(np.isfinite(Hs)) and np.all(Hs>0) and np.max(np.abs(np.diff(Hs)))<10.0
    try:
        r = solve_growth(p, zmax=ZMAX, npts=NPTS, sigma8_0=SIGMA8_0)
        Dv = np.asarray(r["D"]); fs = np.asarray(r["fs8"])
        g_ok = np.all(np.isfinite(Dv)) and np.all(np.isfinite(fs)) and np.max(np.abs(Dv))<10.0
        tv = float(np.sum(np.abs(np.diff(Dv))))
    except Exception:
        g_ok, tv, r, fs = False, float("inf"), {"z":[]}, np.array([np.inf])

    # RMS-normalized stress
    jitter = []
    for key in ("A","f","phi","gamma"):
        p2 = dict(p); p2[key] = p[key]*(1.0+PERT_EPS) if key!="phi" else p[key] + (PERT_EPS*math.pi)
        try:
            r2 = solve_growth(p2, zmax=ZMAX, npts=NPTS, sigma8_0=SIGMA8_0)
            J  = np.interp(r["z"], r2["z"], r2["fs8"])
            num = float(np.sqrt(np.mean((J - fs)**2)))
            den = float(np.sqrt(np.mean(fs**2)) + 1e-6)
            jitter.append(num/den)
        except Exception:
            jitter.append(float("inf"))
    stress = float(np.median(jitter)) if jitter else float("inf")
    stable = bool(h_ok and g_ok and (tv < 5.0) and (stress < 0.25))
    return stable, stress, tv, r

def fs8_score(r):
    # amplitude fit (S8-like) and score vs data
    z_pred = np.asarray(r["z"]); y = np.asarray(r["fs8"])
    y_at = np.interp(D["z"].values, z_pred, y)
    d = D["fs8_obs"].values; s = D["sigma"].values
    w = 1.0/(s*s)
    num = float(np.sum(w * d * y_at)); den = float(np.sum(w * y_at * y_at) + 1e-12)
    alpha = num/den
    yfit = alpha * y_at
    resid = (yfit - d)/s
    chi2 = float(np.sum(resid*resid))
    AIC, BIC = aic_bic(chi2, 1, len(D))  # k=1 for alpha
    dA = dB = None
    if ref:
        dA = AIC - ref["AIC"]["value"]
        dB = BIC - ref["BIC"]["value"]
    return alpha, chi2, AIC, BIC, dA, dB, wwi(dA,dB)

rows=[]
for f in F_GRID:
    for g in GAM_GRID:
        p = dict(BASE); p["f"]=float(f); p["gamma"]=float(g)
        stable, stress, tv, r = metric_stability(p)
        alpha, chi2, AIC, BIC, dA, dB, W = fs8_score(r)
        rows.append({
            "f":f,"gamma":g,
            "stable":stable,"stress_relL2":stress,"TV_D":tv,
            "alpha":alpha,"chi2":chi2,"AIC":AIC,"BIC":BIC,
            "dAIC":dA,"dBIC":dB,"WWI":W
        })

os.makedirs("outputs/phase2", exist_ok=True)
pd.DataFrame(rows).to_csv("outputs/phase2/basin_grid.csv", index=False)
print("Wrote outputs/phase2/basin_grid.csv")
