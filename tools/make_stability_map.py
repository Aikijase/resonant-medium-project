import os, json, math, numpy as np, pandas as pd
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from growth import solve_growth, H_of_z

# ---- Grids (tweak if you like) ----
F_GRID   = np.linspace(2.40, 2.85, 19)   # f sweep around your pocket
GAM_GRID = np.linspace(1.40, 1.65, 11)   # γ sweep around ~1.54
ZMAX, NPTS, SIGMA8_0 = 2.0, 200, 0.8
PERT_EPS = 0.01  # 1% jitter for stress sensitivity

# Load base params from TOML
import tomllib
with open("configs/resonant.toml", "rb") as fh:
    BASE = tomllib.load(fh)["params"]

def metric_stability(p):
    """Compute stability metrics for params p."""
    # 0) H(z) sanity
    zs = np.linspace(0, ZMAX, 100)
    Hs = np.array([H_of_z(z, p) for z in zs])
    h_ok = np.all(np.isfinite(Hs)) and np.all(Hs > 0) and np.max(np.abs(np.diff(Hs))) < 10.0

    # 1) Growth solution sanity + smoothness
    try:
        res = solve_growth(p, zmax=ZMAX, npts=NPTS, sigma8_0=SIGMA8_0)
        D = np.asarray(res["D"]); fs8 = np.asarray(res["fs8"])
        g_ok = np.all(np.isfinite(D)) and np.all(np.isfinite(fs8)) and np.max(np.abs(D)) < 10.0
        tv = float(np.sum(np.abs(np.diff(D))))  # total variation as a ringing proxy
    except Exception:
        g_ok, tv, D = False, float("inf"), None

    # 2) Stress test: sensitivity of fσ8 to tiny jitters (RMS-normalized)
    jitter = []
    for key in ("A","f","phi","gamma"):
        p2 = dict(p)
        p2[key] = p[key]*(1.0+PERT_EPS) if key!="phi" else p[key] + (PERT_EPS*math.pi)
        try:
            r1 = solve_growth(p,  zmax=ZMAX, npts=NPTS, sigma8_0=SIGMA8_0)
            r2 = solve_growth(p2, zmax=ZMAX, npts=NPTS, sigma8_0=SIGMA8_0)
            # L2 over common grid
            J   = np.interp(r1["z"], r2["z"], r2["fs8"])
            diff = J - r1["fs8"]
            num  = np.sqrt(np.mean(diff**2))
            den  = np.sqrt(np.mean(r1["fs8"]**2)) + 1e-6   # RMS of signal
            rel  = num / den
            jitter.append(rel)
        except Exception:
            jitter.append(np.inf)
    stress = float(np.median(jitter)) if jitter else float("inf")

    stable = bool(h_ok and g_ok and (tv < 5.0) and (stress < 0.25))
    return {
        "H_min": float(np.min(Hs)) if Hs.size else float("nan"),
        "D_max": float(np.max(np.abs(D))) if D is not None else float("nan"),
        "TV_D":  tv,
        "stress_relL2": stress,
        "stable": stable
    }

rows = []
for f in F_GRID:
    for g in GAM_GRID:
        p = dict(BASE); p["f"] = float(f); p["gamma"] = float(g)
        m = metric_stability(p)
        rows.append({"f": f, "gamma": g, **m})

os.makedirs("outputs/phase2", exist_ok=True)
outcsv = "outputs/phase2/stability_map.csv"
pd.DataFrame(rows).to_csv(outcsv, index=False)
print("Wrote", outcsv)
