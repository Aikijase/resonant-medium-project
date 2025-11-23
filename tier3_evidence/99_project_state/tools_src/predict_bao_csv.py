import argparse, json, math, pathlib
import numpy as np
import pandas as pd

def load_cov(path: str) -> np.ndarray:
    with open(path, "r") as f:
        s = f.read(4096)
    delim = "," if s.count(",") > s.count(" ") else None  # None => whitespace
    return np.genfromtxt(path, delimiter=delim)

# --- TODO: wire to your project's actual BAO observable function.
# Try uncommenting the import below if you already expose one:
# from joint_fit_resonant_anchors import dm_over_rd_model as resonant_core

def resonant_DM_over_rd(z: np.ndarray, bg: dict, p: dict) -> np.ndarray:
    """
    Return the BAO observable y_model (e.g., DM/rd or DH/rd) for each z.
    Replace the placeholder with your real model call (same one used in fitting).
    """
    # Example if your fitter exposes a function:
    # return resonant_core(z, bg["H0"], bg["Om"], bg["rd"], p["A"], p["f"], p["phi"], p["gamma"])

    # --- PLACEHOLDER (replace me with real physics) ---
    return np.ones_like(z, dtype=float)

def chi2(y: np.ndarray, y_model: np.ndarray, C: np.ndarray) -> float:
    r = y - y_model
    try:
        L = np.linalg.cholesky(C)
        x = np.linalg.solve(L, r)
        return float(x @ x)
    except np.linalg.LinAlgError:
        Ci = np.linalg.pinv(C)
        return float(r @ Ci @ r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True, help="JSON with frozen params")
    ap.add_argument("--bao-csv", required=True)
    ap.add_argument("--bao-cov", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    F = json.load(open(args.freeze))
    bg     = F["background"]
    res_p  = F["resonant_params"]
    dmu    = F["nuisance"].get("dmu", 0.0)

    # ΛCDM baseline uses same bg + same nuisance, but resonance off:
    lcdm_p = {"A":0.0, "f":1.0, "phi":0.0, "gamma":0.0}

    df = pd.read_csv(args.bao_csv)
    if "y" not in df.columns and "y_data" in df.columns:
        df = df.rename(columns={"y_data":"y"})
    for c in ("z","y"):
        if c not in df.columns:
            raise SystemExit(f"Missing column '{c}' in {args.bao_csv}. Got {list(df.columns)}")

    z = df["z"].astype(float).to_numpy()
    y = df["y"].astype(float).to_numpy()

    C = load_cov(args.bao_cov)
    if C.shape != (len(y), len(y)):
        raise SystemExit(f"Covariance shape {C.shape} incompatible with vector length {len(y)}")

    # Build model vectors; if your observable is in distance units, apply dμ here if appropriate.
    # For compressed BAO ratios (DM/rd, DH/rd), dμ typically does NOT enter BAO directly.
    y_res  = resonant_DM_over_rd(z, bg, res_p)
    y_lcdm = resonant_DM_over_rd(z, bg, lcdm_p)

    # χ² and information criteria (no fitted params here -> k=0)
    k = 0
    N = len(y)
    chi_res  = chi2(y, y_res,  C)
    chi_lcdm = chi2(y, y_lcdm, C)

    AIC_res = chi_res  + 2*k
    AIC_lcdm= chi_lcdm + 2*k
    BIC_res = chi_res  + k*math.log(N)
    BIC_lcdm= chi_lcdm + k*math.log(N)

    out = {
        "N": N,
        "chi2": {"res": chi_res, "lcdm": chi_lcdm, "delta": chi_res-chi_lcdm},
        "AIC":  {"res": AIC_res, "lcdm": AIC_lcdm, "delta": AIC_res-AIC_lcdm},
        "BIC":  {"res": BIC_res, "lcdm": BIC_lcdm, "delta": BIC_res-BIC_lcdm},
        "freeze": F
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
