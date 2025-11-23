#!/usr/bin/env python3
"""
Resonant BAO+SN joint fit — auto-adapting, self-healing runner.

- Loads Pantheon+ SN (vector + full covariance) and DESI DR1 BAO long vector + covariance.
- Adapts to whatever function signatures exist in joint_fit_resonant_rd.py:
    - make_sn_chi2(...)
    - make_bao_chi2_both(...)
- Auto-rescales BOTH BAO and SN covariances so full-cov χ² matches diag-only scale at θ0.
- Optimizes [A, f, phi, gamma] with SciPy and writes <out-prefix>.json.
"""
from pathlib import Path
import sys, json, math, argparse, time, inspect
import numpy as np
import pandas as pd

# Make project root importable when running from tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fit_resonant_rd as J   # we adapt to its function signatures

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Auto-adapting BAO+SN joint fit runner.")
    p.add_argument("--bao-csv", required=True)
    p.add_argument("--bao-cov", required=True)
    p.add_argument("--sn-csv",  required=True)
    p.add_argument("--sn-cov",  required=True)
    p.add_argument("--out-prefix", required=True)

    p.add_argument("--assume-per-rd", action="store_true",
                   help="Use DESI long vector as /rd basis (typical for DR1).")
    p.add_argument("--plus-lya", action="store_true", help="Keep Lyα entries if present.")
    p.add_argument("--diag-only", action="store_true", help="Ignore off-diagonals (smoke test).")

    p.add_argument("--H0", type=float, default=70.0)
    p.add_argument("--Om", type=float, default=0.3)
    p.add_argument("--rd", type=float, default=147.1)

    p.add_argument("--optimizer", default="lbfgsb", choices=["lbfgsb","nelder","powell"])
    p.add_argument("--max-evals", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)

    # Initial guess for [A, f, phi, gamma]
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
    Heuristic: if mean diag is >> 1, likely precision.
    """
    md = float(np.mean(np.diag(M)))
    looks_prec = md > 1.0
    try:
        L = np.linalg.cholesky(M)
        return (L, looks_prec) if looks_prec else (L, False)
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

    # Use DESI long vector column explicitly
    if "y_data" in df.columns:
        y = df["y_data"].to_numpy(float)
    else:
        # Fallbacks if a different export is used
        for col in ["y","obs","value","meas","x"]:
            if col in df.columns:
                y = df[col].to_numpy(float)
                break
        else:
            # last numeric column (very last resort)
            y = df.select_dtypes(include=[np.number]).iloc[:,-1].to_numpy(float)

    C = _load_cov(bao_cov_csv, diag_only)
    if C.shape[0] != y.shape[0]:
        raise ValueError(f"BAO cov shape {C.shape} != BAO vector length {y.shape[0]}")

    # Factorize like SN so χ² builder can use either C or factors
    F, is_prec = _factorize(C)
    sigma = df["sigma"].to_numpy(float) if "sigma" in df.columns else None

    return {
        "y": y, "y_data": y, "obs": y, "value": y, "meas": y, "x": y,
        "C": C, "cov": C,
        "F": F, "L": F, "R": F, "L_or_R": F,
        "is_prec": is_prec,
        "sigma": sigma
    }


# ---------- dynamic adapters ----------
def call_with_context(fn, context):
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in context:
            kwargs[name] = context[name]
        else:
            if param.default is inspect._empty:
                # leave missing; if truly required, Python will raise
                pass
    return fn(**kwargs)

def call_make_sn_chi2(sn_pack, H0, Om):
    ctx = dict(sn_pack)
    ctx.update({"sn_pack": sn_pack, "H0": H0, "Om": Om, "Om0": Om})
    return call_with_context(J.make_sn_chi2, ctx)

def call_make_bao_chi2(bao_pack, H0, Om, rd, assume_per_rd, plus_lya):
    ctx = dict(bao_pack)
    ctx.update({
        "bao_pack": bao_pack, "H0": H0, "Om": Om, "Om0": Om,
        "rd": rd, "rd0": rd,
        "assume_per_rd": assume_per_rd, "plus_lya": plus_lya,
        "fit_Om": False, "fit_rd": False, "nz_bg": 800
    })
    return call_with_context(J.make_bao_chi2_both, ctx)

# ---------- objective & optimizer ----------
def _diag_cov(C): return np.diag(np.diag(C))

def _rescale_pack_cov_for_fn(pack, make_fn, theta0, *fn_args):
    """Return (scale, fn_full, fn_diag). If scale!=1, pack['C'] is replaced by C/scale."""
    C_full = pack["C"]
    fn_full = make_fn(pack, *fn_args)
    chi2_full = float(fn_full(theta0))

    pack_diag = dict(pack)
    pack_diag["C"] = _diag_cov(C_full)
    fn_diag = make_fn(pack_diag, *fn_args)
    chi2_diag = float(fn_diag(theta0))

    scale = 1.0
    if chi2_diag > 0:
        s = chi2_full / chi2_diag
        if np.isfinite(s) and s > 0:
            pack["C"] = C_full / s
            scale = s
            fn_full = make_fn(pack, *fn_args)  # rebuild with rescaled C
    return scale, fn_full, fn_diag, chi2_full, chi2_diag

def make_objective(args):
    sn_pack  = load_sn_pack(args.sn_csv,  args.sn_cov,  args.diag_only)
    bao_pack = load_bao_pack(args.bao_csv, args.bao_cov, args.diag_only)

    # χ² callables from your module (initial)
    chi2_sn  = call_make_sn_chi2(sn_pack, args.H0, args.Om)
    chi2_bao = call_make_bao_chi2(bao_pack, args.H0, args.Om, args.rd,
                                  args.assume_per_rd, args.plus_lya)

    def chi2(theta): return float(chi2_sn(theta) + chi2_bao(theta))

    # --- Auto-rescale BOTH covariances (only when using full C) ---
    if not args.diag_only:
        x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], dtype=float)

        # BAO rescale
        s_bao, chi2_bao, chi2_bao_diag, cfull_bao, cdiag_bao = _rescale_pack_cov_for_fn(
            bao_pack, lambda pack, H0, Om, rd, aprd, lya: call_make_bao_chi2(
                pack, H0, Om, rd, aprd, lya),
            x0, args.H0, args.Om, args.rd, args.assume_per_rd, args.plus_lya
        )
        if s_bao != 1.0:
            print(f"[AUTO] Rescaled BAO C by 1/{s_bao:.3e} (full@θ0={cfull_bao:.3e}, diag@θ0={cdiag_bao:.3e})",
                  flush=True)

        # SN rescale
        s_sn, chi2_sn, chi2_sn_diag, cfull_sn, cdiag_sn = _rescale_pack_cov_for_fn(
            sn_pack, lambda pack, H0, Om: call_make_sn_chi2(pack, H0, Om),
            x0, args.H0, args.Om
        )
        if s_sn != 1.0:
            print(f"[AUTO] Rescaled SN  C by 1/{s_sn:.3e} (full@θ0={cfull_sn:.3e}, diag@θ0={cdiag_sn:.3e})",
                  flush=True)

        def chi2(theta): return float(chi2_sn(theta) + chi2_bao(theta))

    return chi2, chi2_sn, chi2_bao, sn_pack, bao_pack

def minimize(chi2, x0, method, max_evals, seed):
    import scipy.optimize as opt
    if method == "lbfgsb":
        bounds = [(0.0, 5.0), (0.0, 8.0), (-math.pi, math.pi), (0.0, 2.0)]
        return opt.minimize(chi2, x0, method="L-BFGS-B",
                            bounds=bounds, options={"maxfun": max_evals})
    if method == "nelder":
        return opt.minimize(chi2, x0, method="Nelder-Mead",
                            options={"maxfev": max_evals})
    return opt.minimize(chi2, x0, method="Powell",
                        options={"maxfev": max_evals})

# ---------- main ----------
def main():
    args = parse_args()
    chi2, chi2_sn, chi2_bao, sn_pack, bao_pack = make_objective(args)

    x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], dtype=float)
    t0 = time.time()
    res = minimize(chi2, x0, args.optimizer, args.max_evals, args.seed)
    t1 = time.time()

    A,f,phi,gamma = [float(v) for v in res.x]
    chi2_min = float(res.fun)

    chi2_sn_best  = float(chi2_sn(res.x))
    chi2_bao_best = float(chi2_bao(res.x))

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

