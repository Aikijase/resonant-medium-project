#!/usr/bin/env python3
import os, sys, json, math, glob
import numpy as np
import pandas as pd

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Reuse the resonant potential proxy from your ISW module
from isw import phi_proxy as phi_proxy_res

# -------- Vectorized LCDM baseline proxies --------
def H_lcdm(z, H0=70.0, Om=0.3, Ol=0.7):
    z = np.asarray(z, dtype=float)
    return H0 * np.sqrt(Om*(1.0 + z)**3 + Ol)

def phi_proxy_lcdm(z):
    z = np.asarray(z, dtype=float)
    Hz = H_lcdm(z)
    Hz = np.clip(Hz, 1e-9, None)
    return (1.0 / (Hz**2)) * np.exp(-z)

# -------- Params loader --------
def load_params(toml_path="configs/resonant.toml"):
    import tomllib
    with open(toml_path, "rb") as fh:
        return tomllib.load(fh)["params"]

# -------- Robust, shape-only lensing amplitude --------
# We use a simple broad kernel 0 <= z <= ZMAX to emulate the CMB lensing kernel support.
ZMAX = 5.0
NZ   = 1024

def winsorize(x, p_lo=5, p_hi=95):
    lo, hi = np.percentile(x, p_lo), np.percentile(x, p_hi)
    return np.clip(x, max(lo, 1e-12), hi)

def predict_A_L(P, zmax=ZMAX, nz=NZ):
    z = np.linspace(0.0, zmax, int(nz))
    # Proxies over the line of sight
    res = np.asarray(phi_proxy_res(z, P), dtype=float)
    lcd = phi_proxy_lcdm(z).astype(float)

    if (not np.all(np.isfinite(res))) or (not np.all(np.isfinite(lcd))):
        return np.nan
    if np.any(lcd <= 0):
        return np.nan

    # Lensing power ∝ ⟨Φ^2⟩ effective; use shape-only normalization
    R2 = winsorize(res**2)
    L2 = winsorize(lcd**2)

    mR, mL = np.median(R2), np.median(L2)
    if not np.isfinite(mR) or not np.isfinite(mL) or mR <= 0 or mL <= 0:
        return np.nan

    # Pointwise ratio of normalized fields, clipped to plausible range
    ratio = (R2/mR) / (L2/mL)
    ratio = np.clip(ratio, 0.5, 1.5)

    # Average across z (uniform weight; could add simple lensing-like weight later)
    return float(np.mean(ratio))

# -------- Scoring helpers --------
def aic_bic(chi2, k, n):
    AIC = chi2 + 2*k
    BIC = chi2 + k*math.log(max(n,1))
    return AIC, BIC

def wwi(dA, dB):
    if dA is None or dB is None or not np.isfinite(dA) or not np.isfinite(dB):
        return None
    return 100 * min(1.0, max(0.0, -dA/10.0), max(0.0, -dB/10.0))

# -------- Main --------
if __name__ == "__main__":
    P = load_params()

    # Load amplitude-format CSV(s) from data/cmb/
    os.makedirs("data/cmb", exist_ok=True)
    paths = sorted(glob.glob("data/cmb/*.csv"))

    # If none found, create a default Planck 2018 Alens row
    default_csv = "data/cmb/lensing_amp.csv"
    if not paths:
        with open(default_csv, "w") as fh:
            fh.write("survey,A_L,sigma_A,notes\n")
            fh.write("PLANCK_2018,1.00,0.03,Planck lensing amplitude\n")
        paths = [default_csv]

    rows = []
    required = {"survey","A_L","sigma_A"}
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not required.issubset(set(df.columns)):
            continue
        for _, r in df.iterrows():
            try:
                rows.append({
                    "survey":   str(r["survey"]),
                    "A_L":      float(r["A_L"]),
                    "sigma_A":  float(r["sigma_A"]),
                    "notes":    str(r.get("notes","")),
                })
            except Exception:
                continue

    if not rows:
        raise SystemExit("No lensing amplitude rows found under data/cmb/*.csv")

    # Predict model A_L
    scored = []
    for r in rows:
        A_mod = predict_A_L(P)
        if not np.isfinite(A_mod):  # skip pathological
            continue
        sigma = max(r["sigma_A"], 1e-9)
        resid = (A_mod - r["A_L"]) / sigma
        scored.append({**r, "A_model": A_mod, "resid": resid})

    DF = pd.DataFrame(scored)
    if DF.empty:
        raise SystemExit("No valid lensing rows after modeling (check inputs)")

    # Combined χ²/AIC/BIC (k=0, no nuisance)
    chi2 = float(np.sum(DF["resid"]**2))
    n = int(len(DF)); k = 0
    AIC, BIC = aic_bic(chi2, k, n)

    # Reference: LCDM A_L = 1
    chi2_ref = float(np.sum(((1.0 - DF["A_L"]) / DF["sigma_A"])**2))
    AIC_ref, BIC_ref = aic_bic(chi2_ref, 0, n)
    dA, dB = AIC - AIC_ref, BIC - BIC_ref
    WWI = wwi(dA, dB)

    # Per-survey breakdown
    groups = []
    for sv, sub in DF.groupby("survey"):
        chi2_s = float(np.sum(sub["resid"]**2)); ns = int(len(sub)); ks = 0
        AIC_s, BIC_s = aic_bic(chi2_s, ks, ns)
        chi2_ref_s = float(np.sum(((1.0 - sub["A_L"]) / sub["sigma_A"])**2))
        AIC_ref_s, BIC_ref_s = aic_bic(chi2_ref_s, 0, ns)
        groups.append({
            "survey": sv,
            "n": ns, "chi2": chi2_s,
            "AIC": AIC_s, "BIC": BIC_s,
            "AIC_ref": AIC_ref_s, "BIC_ref": BIC_ref_s,
            "dAIC": AIC_s - AIC_ref_s, "dBIC": BIC_s - BIC_ref_s,
            "WWI": wwi(AIC_s - AIC_ref_s, BIC_s - BIC_ref_s),
        })

    out = {
        "probe": "cmb_lensing_amp",
        "n": n, "k": k,
        "chi2": chi2,
        "AIC": {"value": AIC, "delta": dA},
        "BIC": {"value": BIC, "delta": dB},
        "WWI": WWI,
        "per_survey": groups,
        "rows": DF.to_dict(orient="records"),
        "reference": {"model": "LCDM A_L=1", "AIC": AIC_ref, "BIC": BIC_ref},
        "z_support": {"zmax": ZMAX, "nz": NZ, "note": "shape-only Φ^2 proxy"},
    }

    os.makedirs("outputs/phase4", exist_ok=True)
    with open("outputs/phase4/lensing_score.json","w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote outputs/phase4/lensing_score.json  (n={n}, chi2={chi2:.3f}, WWI={'None' if WWI is None else f'{WWI:.1f}'})")
