#!/usr/bin/env python3
"""
Resonant BAO+SN joint fit runner (fixed BAO inputs).

Key fixes vs prior attempt:
- BAO loader now passes z and kind (DM/DH) alongside y, sigma, C.
- Both SN and BAO expose multiple aliases (y/mu, C/cov, L/R/F, etc.).
- Adapts to your joint_fit_resonant_rd.make_sn_chi2 / make_bao_chi2_both via kwargs.
- Auto-rescales BOTH SN and BAO covariances so full-cov χ² ~ diag-only χ² at θ0.
- Prints and saves BAO/SN χ² breakdown.

Run with:
  PYTHONPATH=. python3 -u tools/res_joint_bao_fixed.py \
    --bao-csv data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv \
    --bao-cov data/desi_dr1_bao/bao_covariance_plus_lya.csv \
    --sn-csv  data/pantheon_plus/sn_MATCHED_mu.csv \
    --sn-cov  data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.spd.csv \
    --out-prefix outputs/res_fixed_v1 \
    --assume-per-rd --plus-lya \
    --optimizer lbfgsb --max-evals 250
"""
from pathlib import Path
import sys, json, math, argparse, time, inspect
import numpy as np
import pandas as pd

# Make project root importable when running from tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fit_resonant_rd as J  # uses: make_sn_chi2, make_bao_chi2_both

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="BAO+SN joint fit (passes z/kind to BAO).")
    p.add_argument("--bao-csv", required=True)
    p.add_argument("--bao-cov", required=True)
    p.add_argument("--sn-csv",  required=True)
    p.add_argument("--sn-cov",  required=True)
    p.add_argument("--out-prefix", required=True)

    p.add_argument("--assume-per-rd", action="store_true")
    p.add_argument("--plus-lya", action="store_true")
    p.add_argument("--diag-only", action="store_true")

    p.add_argument("--H0", type=float, default=70.0)
    p.add_argument("--Om", type=float, default=0.3)
    p.add_argument("--rd", type=float, default=147.1)

    p.add_argument("--optimizer", default="lbfgsb", choices=["lbfgsb","nelder","powell"])
    p.add_argument("--max-evals", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)

    # initial guess for [A, f, phi, gamma]
    p.add_argument("--A0", type=float, default=0.4)
    p.add_argument("--f0", type=float, default=2.4)
    p.add_argument("--phi0", type=float, default=1.1)
    p.add_argument("--gamma0", type=float, default=0.0)
    return p.parse_args()

# ---------- loaders ----------
def _load_cov(csv_path: str, diag_only: bool):
    C = np.asarray(pd.read_csv(csv_path, header=None), dtype=float)
    if diag_only:
        C = np.diag(np.diag(C))
    return C

def _factorize(M: np.ndarray):
    """Return (F, is_prec) so that:
       if cov:  F = L  with  L L^T = C,  is_prec=False
       if prec: F = R  with  R^T R = P,  is_prec=True
    Heuristic: large diag -> likely precision; we still try Cholesky either way.
    """
    try:
        L = np.linalg.cholesky(M)
        return L, False
    except np.linalg.LinAlgError:
        R = np.linalg.cholesky(M)
        return R, True

def load_sn_pack(sn_csv: str, sn_cov_csv: str, diag_only: bool):
    df = pd.read_csv(sn_csv)
    z  = df["z"].to_numpy(float)  if "z"  in df.columns else df.iloc[:,0].to_numpy(float)
    mu = df["mu"].to_numpy(float) if "mu" in df.columns else df.iloc[:,1].to_numpy(float)
    C = _load_cov(sn_cov_csv, diag_only)
    if C.shape[0] != mu.shape[0]:
        raise ValueError(f"SN cov shape {C.shape} != SN vector length {mu.shape[0]}")
    F, is_prec = _factorize(C)
    return {
        "z": z, "mu": mu, "y": mu,
        "C": C, "cov": C,
        "F": F, "L": F, "R": F, "L_or_R": F,
        "is_prec": is_prec
    }

def load_bao_pack(bao_csv: str, bao_cov_csv: str, diag_only: bool):
    df = pd.read_csv(bao_csv)
    # Your file has columns: kind, z, y_data, sigma
    if not {"kind","z"}.issubset(df.columns):
        raise ValueError("BAO CSV must include 'kind' and 'z' columns.")
    if "y_data" in df.columns:
        y = df["y_data"].to_numpy(float)
    else:
        # fallback to any typical name
        for col in ["y","obs","value","meas","x"]:
            if col in df.columns:
                y = df[col].to_numpy(float); break
        else:
            y = df.select_dtypes(include=[np.number]).iloc[:,-1].to_numpy(float)

    z     = df["z"].to_numpy(float)
    kind  = df["kind"].astype(str).to_numpy()
    sigma = df["sigma"].to_numpy(float) if "sigma" in df.columns else None

    C = _load_cov(bao_cov_csv, diag_only)
    if C.shape[0] != y.shape[0]:
        raise ValueError(f"BAO cov shape {C.shape} != BAO vector length {y.shape[0]}")

    F, is_prec = _factorize(C)
    return {
        "y": y, "y_data": y, "obs": y, "value": y, "meas": y, "x": y,
        "z": z, "kind": kind,
        "C": C, "cov": C,
        "F": F, "L": F, "R": F, "L_or_R": F,
        "is_prec": is_prec,
        "sigma": sigma
    }

# ---------- adapters ----------
def _call_with_context(fn, ctx):
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ctx:
            kwargs[name] = ctx[name]
        elif param.default is inspect._empty:
            # leave missing; if truly required, Python will raise
            pass
    return fn(**kwargs)

def make_sn_fun(sn_pack, H0, Om):
    ctx = {**sn_pack, "sn_pack": sn_pack, "H0": H0, "Om": Om, "Om0": Om}
    return _call_with_context(J.make_sn_chi2, ctx)

def make_bao_fun(bao_pack, H0, Om, rd, assume_per_rd, plus_lya):
    ctx = {
        **bao_pack,
        "bao_pack": bao_pack,
        "H0": H0, "Om": Om, "Om0": Om,
        "rd": rd, "rd0": rd,
        "assume_per_rd": assume_per_rd,
        "plus_lya": plus_lya,
        "fit_Om": False, "fit_rd": False, "nz_bg": 800
    }
    return _call_with_context(J.make_bao_chi2_both, ctx)

# ---------- objective / auto-rescale ----------
def _diag_cov(C): return np.diag(np.diag(C))

def _rescale_cov_for(fn_maker, pack, theta0, *fn_args):
    """Rescale pack['C'] AND its factor (L_or_R/F) so full-cov χ² at θ0
       matches diag-only χ². Returns (scale, fn_full, fn_diag, chi2_full, chi2_diag).
    """
    C_full = pack["C"]
    f_full = fn_maker(pack, *fn_args)
    chi2_full = float(f_full(theta0))

    pack_d = dict(pack); pack_d["C"] = np.diag(np.diag(C_full))
    # also rebuild a diagonal factor for the diag case (covariance)
    if "F" in pack_d:
        # If original was precision, we still want a cov-style diag factor here
        Fd = np.linalg.cholesky(pack_d["C"])
        for k in ("F","L","R","L_or_R"):
            pack_d[k] = Fd
        pack_d["is_prec"] = False

    f_diag = fn_maker(pack_d, *fn_args)
    chi2_diag = float(f_diag(theta0))

    scale = 1.0
    if chi2_diag > 0:
        s = chi2_full / chi2_diag
        if np.isfinite(s) and s > 0:
            # Rescale covariance
            pack["C"] = C_full / s
            # Rescale factor depending on whether original pack looked like cov or precision
            is_prec = bool(pack.get("is_prec", False))
            F = pack.get("L_or_R", pack.get("F", None))
            if F is not None:
                if is_prec:
                    # R' = sqrt(s) * R  (since P' = s * P)
                    F = (math.sqrt(s)) * F
                else:
                    # L' = L / sqrt(s)  (since C' = C / s)
                    F = F / (math.sqrt(s))
                for k in ("F","L","R","L_or_R"):
                    pack[k] = F

            # Rebuild full χ² with rescaled pack
            f_full = fn_maker(pack, *fn_args)
            scale = s

    return scale, f_full, f_diag, chi2_full, chi2_diag

# ---------- optimize ----------
def minimize(chi2, x0, method, max_evals, seed):
    import scipy.optimize as opt
    if method == "lbfgsb":
        bounds = [(0.0, 5.0), (0.0, 8.0), (-math.pi, math.pi), (0.0, 2.0)]
        return opt.minimize(chi2, x0, method="L-BFGS-B",
                            bounds=bounds, options={"maxfun": max_evals})
    if method == "nelder":
        return opt.minimize(chi2, x0, method="Nelder-Mead",
                            options={"maxfev": max_evals})
    return opt.minimize(chi2, x0, method="Powell", options={"maxfev": max_evals})

# ---------- main ----------
def main():
    args = parse_args()

    sn_pack  = load_sn_pack(args.sn_csv,  args.sn_cov,  args.diag_only)
    bao_pack = load_bao_pack(args.bao_csv, args.bao_cov, args.diag_only)

    sn_fun  = make_sn_fun(sn_pack, args.H0, args.Om)
    bao_fun = make_bao_fun(bao_pack, args.H0, args.Om, args.rd,
                           args.assume_per_rd, args.plus_lya)

    def chi2(theta): return float(sn_fun(theta) + bao_fun(theta))

    # Auto-rescale both covariances (full-cov only)
    if not args.diag_only:
        x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], float)
        s_bao, bao_fun, bao_fun_d, cfull_bao, cdiag_bao = _rescale_cov_for(
            make_bao_fun, bao_pack, x0, args.H0, args.Om, args.rd,
            args.assume_per_rd, args.plus_lya
        )
        if s_bao != 1.0:
            print(f"[AUTO] Rescaled BAO C by 1/{s_bao:.3e} (full@θ0={cfull_bao:.3e}, diag@θ0={cdiag_bao:.3e})",
                  flush=True)

        s_sn, sn_fun, sn_fun_d, cfull_sn, cdiag_sn = _rescale_cov_for(
            make_sn_fun, sn_pack, x0, args.H0, args.Om
        )
        if s_sn != 1.0:
            print(f"[AUTO] Rescaled SN  C by 1/{s_sn:.3e} (full@θ0={cfull_sn:.3e}, diag@θ0={cdiag_sn:.3e})",
                  flush=True)

        def chi2(theta): return float(sn_fun(theta) + bao_fun(theta))

    x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], float)
    t0 = time.time()
    res = minimize(chi2, x0, args.optimizer, args.max_evals, args.seed)
    t1 = time.time()

    A,f,phi,gamma = [float(v) for v in res.x]
    chi2_min = float(res.fun)

    chi2_sn_best  = float(sn_fun(res.x))
    chi2_bao_best = float(bao_fun(res.x))

    N_bao = int(bao_pack["y"].shape[0])
    N_sn  = int(sn_pack["y"].shape[0])
    N     = N_bao + N_sn
    k     = 4
    ndof  = int(N - k)

    out = {
        "best_params": {"A":A, "f":f, "phi":phi, "gamma":gamma,
                        "H0":args.H0, "Om":args.Om, "rd":args.rd},
        "chi2": chi2_min,
        "ndof": ndof,
        "success": bool(res.success),
        "message": str(res.message),
        "elapsed_s": t1 - t0,
        "nfev": int(getattr(res, "nfev", 0)),
        "breakdown": {"chi2_sn": chi2_sn_best, "chi2_bao": chi2_bao_best},
        "meta": {
            "assume_per_rd": args.assume_per_rd,
            "plus_lya": args.plus_lya,
            "diag_only": args.diag_only,
            "optimizer": args.optimizer,
            "max_evals": args.max_evals
        }
    }

    p = Path(args.out_prefix).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {p}")

if __name__ == "__main__":
    main()
