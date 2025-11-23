import argparse, json, math, os
import numpy as np, pandas as pd
from growth import solve_growth

def aic_bic(chi2, k, n):
    AIC = chi2 + 2 * k
    BIC = chi2 + k * math.log(max(n, 1))
    return AIC, BIC

def wwi(dA, dB):
    if dA is None or dB is None:
        return None
    return 100 * min(1.0, max(0.0, -dA/10.0), max(0.0, -dB/10.0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="cfg", required=True, help="configs/resonant.toml")
    ap.add_argument("--data", default="data/growth/fs8_catalog.csv")
    ap.add_argument("--out", dest="out", required=True, help="outputs/phase2/fs8_eval.json")
    ap.add_argument("--ref", dest="ref", default="", help="LCDM reference JSON for deltas")
    ap.add_argument("--fit-s8", action="store_true", help="fit a global amplitude alpha to fs8_pred")
    args = ap.parse_args()

    import tomllib
    with open(args.cfg, "rb") as fh:
        P = tomllib.load(fh)["params"]

    # Model prediction
    sol = solve_growth(P)
    z_pred = np.asarray(sol["z"])
    fs8_pred = np.asarray(sol["fs8"])

    # Data
    D = pd.read_csv(args.data)
    z_d = D["z"].values
    d = D["fs8_obs"].values
    s = D["sigma"].values
    y = np.interp(z_d, z_pred, fs8_pred)

    # Optional amplitude fit (S8-like)
    alpha, k = 1.0, 0
    if args.fit_s8:
        w = 1.0 / (s * s)
        num = float(np.sum(w * d * y))
        den = float(np.sum(w * y * y) + 1e-12)
        alpha = num / den
        k = 1
    yfit = alpha * y

    # Score
    chi2 = float(np.sum(((yfit - d) / s) ** 2))
    n = int(len(D))
    AIC, BIC = aic_bic(chi2, k, n)

    result = {
        "probe": "fs8",
        "n": n,
        "k": k,
        "chi2": chi2,
        "alpha": alpha if args.fit_s8 else None,
        "AIC": {"value": AIC, "delta": None},
        "BIC": {"value": BIC, "delta": None},
        "WWI": None,
    }

    # Deltas/WWI if reference provided
    if args.ref:
        try:
            R = json.load(open(args.ref))
            dA = AIC - R["AIC"]["value"]
            dB = BIC - R["BIC"]["value"]
            result["AIC"]["delta"] = dA
            result["BIC"]["delta"] = dB
            result["WWI"] = wwi(dA, dB)
        except Exception:
            pass

    # Write outputs
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_csv = args.out.replace(".json", ".csv")
    pd.DataFrame({"z": z_pred, "fs8_pred": fs8_pred}).assign(fs8_fit=np.interp(z_pred, z_pred, alpha*fs8_pred)).to_csv(out_csv, index=False)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out} and {out_csv}")
