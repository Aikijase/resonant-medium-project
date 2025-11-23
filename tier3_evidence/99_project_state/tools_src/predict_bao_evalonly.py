import argparse, json, math, inspect
import numpy as np
import pandas as pd
import sys, pathlib

# Ensure repo root is on sys.path (so we can import joint_fit_resonant_anchors.py)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import joint_fit_resonant_anchors as J


def load_cov(path: str) -> np.ndarray:
    with open(path, "r") as f:
        s = f.read(4096)
    delim = "," if s.count(",") > s.count(" ") else None  # None => whitespace
    return np.genfromtxt(path, delimiter=delim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--bao-csv", required=True)
    ap.add_argument("--bao-cov", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load frozen params
    F = json.load(open(args.freeze))
    bg   = dict(F["background"])
    res  = dict(F["resonant_params"])
    lcdm = {"A": 0.0, "f": 1.0, "phi": 0.0, "gamma": 0.0}

    # Same background on both models
    for k in ("H0", "Om", "rd"):
        res[k] = bg[k]
        lcdm[k] = bg[k]

    # Ensure required anchor keys present on both (BAO chi2 expects anchor_eps)
    eps = res.get("anchor_eps", 0.0)
    zR  = res.get("anchor_zR", 0.5)
    res["anchor_eps"] = eps
    lcdm["anchor_eps"] = eps  # same value in both for fair comparison

    # Load BAO data via project loader
    bao_pack = J.load_bao(args.bao_csv, args.bao_cov)

    # Build kwargs for make_bao_chi2 based on its signature
    sig = inspect.signature(J.make_bao_chi2)
    kws = {}
    if "bao_kind" in sig.parameters:
        kws["bao_kind"] = "dm_over_rd"
    if "H0" in sig.parameters:
        kws["H0"] = bg["H0"]
    if "anchor_zR" in sig.parameters:
        kws["anchor_zR"] = zR
    if "prior_A_sigma" in sig.parameters:
        kws["prior_A_sigma"] = 0.0
    if "assume_per_rd" in sig.parameters:
        kws["assume_per_rd"] = True
    if "plus_lya" in sig.parameters:
        kws["plus_lya"] = False

    bao_chi2 = J.make_bao_chi2(bao_pack, **kws)

    # Evaluate chi2 for frozen params (no fitting; k=0)
    chi_res = float(bao_chi2(res))
    chi_lcdm = float(bao_chi2(lcdm))

    # N and information criteria (k=0 so AIC=BIC=chi2 here)
    N = len(pd.read_csv(args.bao_csv))
    k = 0
    AIC_res, AIC_lcdm = chi_res + 2 * k, chi_lcdm + 2 * k
    BIC_res, BIC_lcdm = chi_res + k * math.log(N), chi_lcdm + k * math.log(N)

    out = {
        "N": N,
        "chi2": {"res": chi_res, "lcdm": chi_lcdm, "delta": chi_res - chi_lcdm},
        "AIC": {"res": AIC_res, "lcdm": AIC_lcdm, "delta": AIC_res - AIC_lcdm},
        "BIC": {"res": BIC_res, "lcdm": BIC_lcdm, "delta": BIC_res - BIC_lcdm},
        "freeze": F,
        "factory_kwargs_used": kws,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
