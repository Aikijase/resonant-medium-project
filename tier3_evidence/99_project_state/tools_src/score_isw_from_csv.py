import os, sys, json, math, glob
import numpy as np
import pandas as pd

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Reuse your resonant proxy
from isw import phi_proxy as phi_proxy_res

# --- Vectorized LCDM baseline ---
def H_lcdm(z, H0=70.0, Om=0.3, Ol=0.7):
    z = np.asarray(z, dtype=float)
    return H0 * np.sqrt(Om*(1.0 + z)**3 + Ol)

def phi_proxy_lcdm(z):
    z = np.asarray(z, dtype=float)
    Hz = H_lcdm(z)
    Hz = np.clip(Hz, 1e-9, None)
    return (1.0 / (Hz**2)) * np.exp(-z)

# --- Load model params from your toml ---
def load_params(toml_path="configs/resonant.toml"):
    import tomllib
    with open(toml_path, "rb") as fh:
        return tomllib.load(fh)["params"]

def predict_A_model(P, zc, halfw, nz=128):
    z1, z2 = max(0.0, float(zc - halfw)), float(zc + halfw)
    if not np.isfinite(z1) or not np.isfinite(z2) or z2 <= z1:
        return np.nan
    z = np.linspace(z1, z2, int(nz))

    # Proxies on the same grid
    res = np.asarray(phi_proxy_res(z, P), dtype=float)
    lcd = phi_proxy_lcdm(z).astype(float)

    if (not np.all(np.isfinite(res))) or (not np.all(np.isfinite(lcd))):
        return np.nan
    if np.any(lcd <= 0):
        return np.nan

    # Winsorize (5–95%) to kill spikes, then normalize by median (shape-only)
    def winsorize(x):
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        return np.clip(x, max(lo, 1e-12), hi)

    res_w = winsorize(res)
    lcd_w = winsorize(lcd)
    med_res = np.median(res_w)
    med_lcd = np.median(lcd_w)
    if not np.isfinite(med_res) or not np.isfinite(med_lcd) or med_res <= 0 or med_lcd <= 0:
        return np.nan

    r = (res_w/med_res) / (lcd_w/med_lcd)   # pointwise ratio after normalization
    r = np.clip(r, 0.2, 2.0)                # keep within plausible amplitude range
    return float(np.mean(r))

# --- Scoring helpers ---
def aic_bic(chi2, k, n):
    AIC = chi2 + 2*k
    BIC = chi2 + k*math.log(max(n,1))
    return AIC, BIC

def wwi(dA, dB):
    if dA is None or dB is None or not np.isfinite(dA) or not np.isfinite(dB):
        return None
    return 100 * min(1.0, max(0.0, -dA/10.0), max(0.0, -dB/10.0))

# Redshift half-width heuristics per survey (edit as needed)
SURVEY_HALFWIDTH = {
    "WISE_GAL":     0.15,
    "WISE_AGN":     0.30,
    "RACS_SKADS":   0.25,
    "RACS_BACCUS":  0.25,
}
DEFAULT_HALFWIDTH = 0.25

if __name__ == "__main__":
    P = load_params()

    # Load amplitude-format CSVs only
    rows = []
    required = {"survey","ell_min","ell_max","A_ISW","sigma_A","z_eff"}
    for path in sorted(glob.glob("data/isw/*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not required.issubset(set(df.columns)):
            continue  # skip non-conforming files
        for _, r in df.iterrows():
            try:
                rows.append({
                    "survey":   str(r["survey"]),
                    "ell_min":  int(r["ell_min"]),
                    "ell_max":  int(r["ell_max"]),
                    "A_ISW":    float(r["A_ISW"]),
                    "sigma_A":  float(r["sigma_A"]),
                    "z_eff":    float(r["z_eff"]),
                    "notes":    str(r.get("notes","")),
                })
            except Exception:
                continue

    if not rows:
        raise SystemExit("No amplitude-format ISW rows found under data/isw/*.csv")

    # Predict A_model and residuals
    scored = []
    for r in rows:
        zc = r["z_eff"]
        hw = SURVEY_HALFWIDTH.get(r["survey"], DEFAULT_HALFWIDTH)
        A_mod = predict_A_model(P, zc, hw)
        if not np.isfinite(A_mod):
            continue
        sigma = max(r["sigma_A"], 1e-9)
        resid = (A_mod - r["A_ISW"]) / sigma
        scored.append({**r, "halfwidth": hw, "A_model": A_mod, "resid": resid})

    DF = pd.DataFrame(scored)
    if DF.empty:
        raise SystemExit("No valid ISW rows after modeling (check inputs/halfwidths)")

    # Combined χ²/AIC/BIC
    chi2 = float(np.sum(DF["resid"]**2))
    n    = int(len(DF))
    k    = 0
    AIC, BIC = aic_bic(chi2, k, n)

    # Reference: LCDM A=1 for all rows
    chi2_ref = float(np.sum(((1.0 - DF["A_ISW"]) / DF["sigma_A"])**2))
    AIC_ref, BIC_ref = aic_bic(chi2_ref, 0, n)
    dA, dB = AIC - AIC_ref, BIC - BIC_ref
    WWI = wwi(dA, dB)

    # Per-survey breakdown
    groups = []
    for sv, sub in DF.groupby("survey"):
        chi2_s = float(np.sum(sub["resid"]**2)); ns = int(len(sub)); ks = 0
        AIC_s, BIC_s = aic_bic(chi2_s, ks, ns)
        chi2_ref_s = float(np.sum(((1.0 - sub["A_ISW"]) / sub["sigma_A"])**2))
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
        "probe": "isw_amplitude",
        "n": n, "k": k,
        "chi2": chi2,
        "AIC": {"value": AIC, "delta": dA},
        "BIC": {"value": BIC, "delta": dB},
        "WWI": WWI,
        "per_survey": groups,
        "rows": DF.to_dict(orient="records"),
        "reference": {"model": "LCDM A=1", "AIC": AIC_ref, "BIC": BIC_ref}
    }

    os.makedirs("outputs/phase2", exist_ok=True)
    with open("outputs/phase2/isw_score.json","w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote outputs/phase2/isw_score.json  (n={n}, chi2={chi2:.3f}, WWI={'None' if WWI is None else f'{WWI:.1f}'})")
