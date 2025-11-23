#!/usr/bin/env python3
"""
Resonant BAO+SN joint fit — smart adapter.

Strategy:
1) Load Pantheon+ SN and DESI DR1 "long" BAO vector (y_data) + z/kind + cov.
2) Create diag-only baseline χ² at θ0 for both SN and BAO.
3) For each of several "full-cov variants" (factor on/off, cov/precision mode):
   - Build χ² function using joint_fit_resonant_rd.*
   - Rescale C (and factor if used) so full-cov χ² @θ0 matches diag χ² @θ0.
   - Score variant by |log(full/diag)|.
4) Pick the lowest-score variant for SN and for BAO.
5) Optimize total χ² = χ²_SN + χ²_BAO with bounds, save JSON (and per-term breakdown).
"""
from pathlib import Path
import sys, json, math, argparse, time, inspect
import numpy as np
import pandas as pd

# import project module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fit_resonant_rd as J

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Smart BAO+SN joint fit runner.")
    p.add_argument("--bao-csv", required=True)
    p.add_argument("--bao-cov", required=True)
    p.add_argument("--sn-csv",  required=True)
    p.add_argument("--sn-cov",  required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--assume-per-rd", action="store_true")  # passed to BAO
    p.add_argument("--plus-lya", action="store_true")       # passed to BAO
    p.add_argument("--diag-only", action="store_true")      # bypass correlations entirely
    p.add_argument("--H0", type=float, default=70.0)
    p.add_argument("--Om", type=float, default=0.3)
    p.add_argument("--rd", type=float, default=147.1)
    p.add_argument("--optimizer", default="lbfgsb", choices=["lbfgsb","nelder","powell"])
    p.add_argument("--max-evals", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--A0", type=float, default=0.4)
    p.add_argument("--f0", type=float, default=2.4)
    p.add_argument("--phi0", type=float, default=1.1)
    p.add_argument("--gamma0", type=float, default=0.0)
    return p.parse_args()

# ---------------- Utils ----------------
def _load_cov(csv_path, diag_only):
    C = np.asarray(pd.read_csv(csv_path, header=None), dtype=float)
    return np.diag(np.diag(C)) if diag_only else C

def _chol_cov(C):
    return np.linalg.cholesky(C)

def _chol_prec(P):
    return np.linalg.cholesky(P)

def _diag(C):
    return np.diag(np.diag(C))

def _make_kwargs(fn, context):
    import numpy as _np
    import numpy.linalg as _la
    sig = inspect.signature(fn)
    ctx = dict(context)  # copy so we can add derived items

    # If the function requires L / R / L_or_R and they're missing, synthesize from C
    need_L = "L" in sig.parameters and "L" not in ctx
    need_R = "R" in sig.parameters and "R" not in ctx
    need_LR = "L_or_R" in sig.parameters and "L_or_R" not in ctx
    if (need_L or need_R or need_LR) and "C" in ctx:
        C = ctx["C"]
        # Try Cholesky of covariance; if that fails, diagonalize as fallback
        try:
            L = _la.cholesky(C)
        except Exception:
            L = _la.cholesky(_np.diag(_np.clip(_np.diag(C), 1e-12, None)))
        # Provide all aliases so whatever the fn wants is present
        ctx.setdefault("L", L)
        ctx.setdefault("R", L)         # if the fn asks for R, we still hand L; most code treats it generically
        ctx.setdefault("L_or_R", L)
        ctx.setdefault("F", L)

    # Build kwargs the fn can accept
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ctx:
            kwargs[name] = ctx[name]
        elif param.default is inspect._empty:
            # leave missing; Python will raise if truly required (helps us discover)
            pass
    return kwargs
def _call(fn, context):
    # Build kwargs from context and call the function
    return fn(**_make_kwargs(fn, context))

# ---------------- Loaders ----------------
def load_sn_pack(sn_csv, sn_cov_csv, diag_only):
    df = pd.read_csv(sn_csv)
    z  = df["z"].to_numpy(float)  if "z" in df.columns else df.iloc[:,0].to_numpy(float)
    mu = df["mu"].to_numpy(float) if "mu" in df.columns else df.iloc[:,1].to_numpy(float)
    C  = _load_cov(sn_cov_csv, diag_only)
    if C.shape[0] != mu.shape[0]:
        raise ValueError(f"SN cov shape {C.shape} != SN vector length {mu.shape[0]}")
    pack = {
        "z": z, "mu": mu, "y": mu,
        "C": C, "cov": C,
        # We'll add factors per-variant as needed
        "is_prec": False  # default: treat as covariance unless variant says otherwise
    }
    return pack

def load_bao_pack(bao_csv, bao_cov_csv, diag_only):
    df = pd.read_csv(bao_csv)
    # Your file: kind, z, y_data, sigma
    if not {"kind","z"}.issubset(df.columns):
        raise ValueError("BAO CSV must include 'kind' and 'z'.")
    y = df["y_data"].to_numpy(float) if "y_data" in df.columns else \
        (df["y"].to_numpy(float) if "y" in df.columns else
         df.select_dtypes(include=[np.number]).iloc[:,-1].to_numpy(float))
    z     = df["z"].to_numpy(float)
    kind  = df["kind"].astype(str).to_numpy()
    sigma = df["sigma"].to_numpy(float) if "sigma" in df.columns else None
    C     = _load_cov(bao_cov_csv, diag_only)
    if C.shape[0] != y.shape[0]:
        raise ValueError(f"BAO cov shape {C.shape} != BAO vector length {y.shape[0]}")
    pack = {
        "y": y, "y_data": y, "obs": y, "value": y, "meas": y, "x": y,
        "z": z, "kind": kind,
        "C": C, "cov": C,
        "sigma": sigma,
        "is_prec": False
    }
    return pack

# ---------------- Variant builder ----------------
def make_sn_variant(pack_base, H0, Om, variant):
    """variant: dict flags
       - use_factor: bool
       - treat_as_prec: bool  (for factor semantics)
    """
    pack = dict(pack_base)
    # factor if asked
    if variant["use_factor"]:
        if variant["treat_as_prec"]:
            # treat C as precision: P, R=chol(P)
            R = _chol_prec(pack["C"])
            pack.update({"L_or_R": R, "F": R, "L": R, "R": R, "is_prec": True})
        else:
            # treat C as covariance: C=L L^T
            L = _chol_cov(pack["C"])
            pack.update({"L_or_R": L, "F": L, "L": L, "R": L, "is_prec": False})
    else:
        # no factor; encourage cov mode
        pack.update({"is_prec": False})
        for k in ("L_or_R","F","L","R"):
            if k in pack: del pack[k]

    fn = J.make_sn_chi2
    ctx = {**pack, "sn_pack": pack, "H0": H0, "Om": Om, "Om0": Om}
    return lambda theta: float(_call(fn, ctx)(theta)), pack

def make_bao_variant(pack_base, H0, Om, rd, assume_per_rd, plus_lya, variant):
    pack = dict(pack_base)
    if variant["use_factor"]:
        if variant["treat_as_prec"]:
            R = _chol_prec(pack["C"])
            pack.update({"L_or_R": R, "F": R, "L": R, "R": R, "is_prec": True})
        else:
            L = _chol_cov(pack["C"])
            pack.update({"L_or_R": L, "F": L, "L": L, "R": L, "is_prec": False})
    else:
        pack.update({"is_prec": False})
        for k in ("L_or_R","F","L","R"):
            if k in pack: del pack[k]

    fn = J.make_bao_chi2_both
    ctx = {
        **pack, "bao_pack": pack,
        "H0": H0, "Om": Om, "Om0": Om,
        "rd": rd, "rd0": rd,
        "assume_per_rd": assume_per_rd, "plus_lya": plus_lya,
        "fit_Om": False, "fit_rd": False, "nz_bg": 800
    }
    return lambda theta: float(_call(fn, ctx)(theta)), pack

def diag_baseline_sn(pack_base, H0, Om):
    pack = dict(pack_base); pack["C"] = _diag(pack["C"])
    L = _chol_cov(pack["C"])
    pack.update({"L_or_R": L, "F": L, "L": L, "R": L, "is_prec": False})
    fn = J.make_sn_chi2
    ctx = {**pack, "sn_pack": pack, "H0": H0, "Om": Om, "Om0": Om}
    return lambda theta: float(_call(fn, ctx)(theta))

def diag_baseline_bao(pack_base, H0, Om, rd, assume_per_rd, plus_lya):
    pack = dict(pack_base); pack["C"] = _diag(pack["C"])
    L = _chol_cov(pack["C"])
    pack.update({"L_or_R": L, "F": L, "L": L, "R": L, "is_prec": False})
    fn = J.make_bao_chi2_both
    ctx = {
        **pack, "bao_pack": pack,
        "H0": H0, "Om": Om, "Om0": Om,
        "rd": rd, "rd0": rd,
        "assume_per_rd": assume_per_rd, "plus_lya": plus_lya,
        "fit_Om": False, "fit_rd": False, "nz_bg": 800
    }
    return lambda theta: float(_call(fn, ctx)(theta))

def pick_variant(pack_base, build_fn, baseline_fn, theta0, rescale=True):
    """Try variants and pick the one whose full-cov χ²(θ0) matches diag baseline best.
       If rescale=True, globally scale C (and factor) to match exactly.
    """
    variants = [
        {"use_factor": True,  "treat_as_prec": False},  # L (cov)
        {"use_factor": True,  "treat_as_prec": True },  # R (prec)
        {"use_factor": False, "treat_as_prec": False},  # no factor, cov
    ]
    best = None
    chi2_base = float(baseline_fn(theta0))
    for v in variants:
        fn, pack = build_fn(pack_base, v)
        c_full = float(fn(theta0))
        # compute scale to match baseline
        scale = 1.0
        if rescale and chi2_base > 0 and np.isfinite(c_full) and c_full > 0:
            s = c_full / chi2_base
            if np.isfinite(s) and s > 0:
                # scale C
                pack["C"] = pack["C"] / s
                # scale factor if present
                if v["use_factor"]:
                    F = pack.get("L_or_R")
                    if F is not None:
                        if v["treat_as_prec"]:
                            # P' = s*P ⇒ R' = sqrt(s) R
                            pack["L_or_R"] = (math.sqrt(s)) * F
                        else:
                            # C' = C/s ⇒ L' = L / sqrt(s)
                            pack["L_or_R"] = F / (math.sqrt(s))
                        for k in ("F","L","R"):
                            pack[k] = pack["L_or_R"]
                # rebuild function with rescaled pack
                fn, pack = build_fn(pack, v)
                c_full = float(fn(theta0))
                scale = s
        score = abs(math.log((c_full/chi2_base) if (chi2_base>0 and c_full>0) else 1e9))
        cand = (score, v, fn, pack, c_full, chi2_base, scale)
        if (best is None) or (score < best[0]):
            best = cand
    return best  # (score, variant, fn, pack, c_full, chi2_base, scale)

# ---------------- Main build ----------------
def build_objective(args):
    sn_base  = load_sn_pack(args.sn_csv,  args.sn_cov,  args.diag_only)
    bao_base = load_bao_pack(args.bao_csv, args.bao_cov, args.diag_only)
    theta0   = np.array([args.A0, args.f0, args.phi0, args.gamma0], float)

    if args.diag_only:
        sn_fn  = diag_baseline_sn(sn_base, args.H0, args.Om)
        bao_fn = diag_baseline_bao(bao_base, args.H0, args.Om, args.rd,
                                   args.assume_per_rd, args.plus_lya)
    else:
        # baselines
        sn_base_fn  = diag_baseline_sn(sn_base, args.H0, args.Om)
        bao_base_fn = diag_baseline_bao(bao_base, args.H0, args.Om, args.rd,
                                        args.assume_per_rd, args.plus_lya)
        # pick best variants (with rescale)
        sn_score,  sn_variant,  sn_fn,  sn_pack,  sn_full0,  sn_diag0,  s_sn  = \
            pick_variant(sn_base,
                         lambda p, v: make_sn_variant(p, args.H0, args.Om, v),
                         sn_base_fn, theta0, rescale=True)
        bao_score, bao_variant, bao_fn, bao_pack, bao_full0, bao_diag0, s_bao = \
            pick_variant(bao_base,
                         lambda p, v: make_bao_variant(p, args.H0, args.Om, args.rd,
                                                       args.assume_per_rd, args.plus_lya, v),
                         bao_base_fn, theta0, rescale=True)
        print(f"[SMART] SN  variant={sn_variant} scale=1/{s_sn:.3e} full@θ0={sn_full0:.3e} diag@θ0={sn_diag0:.3e}", flush=True)
        print(f"[SMART] BAO variant={bao_variant} scale=1/{s_bao:.3e} full@θ0={bao_full0:.3e} diag@θ0={bao_diag0:.3e}", flush=True)

    def chi2(theta): return float(sn_fn(theta) + bao_fn(theta))
    return chi2, sn_fn, bao_fn, sn_base, bao_base

# ---------------- Optimize ----------------
def minimize(chi2, x0, method, max_evals):
    import scipy.optimize as opt
    if method == "lbfgsb":
        bounds = [(0.0, 5.0), (0.0, 8.0), (-math.pi, math.pi), (0.0, 2.0)]
        return opt.minimize(chi2, x0, method="L-BFGS-B",
                            bounds=bounds, options={"maxfun": max_evals})
    if method == "nelder":
        return opt.minimize(chi2, x0, method="Nelder-Mead",
                            options={"maxfev": max_evals})
    return opt.minimize(chi2, x0, method="Powell", options={"maxfev": max_evals})

# ---------------- Entry ----------------
def main():
    args = parse_args()
    chi2, sn_fn, bao_fn, sn_base, bao_base = build_objective(args)
    x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], float)
    t0 = time.time()
    res = minimize(chi2, x0, args.optimizer, args.max_evals)
    t1 = time.time()

    A,f,phi,gamma = map(float, res.x)
    out = {
        "best_params": {"A":A, "f":f, "phi":phi, "gamma":gamma,
                        "H0":args.H0, "Om":args.Om, "rd":args.rd},
        "chi2": float(res.fun),
        "ndof": int(sn_base["C"].shape[0] + bao_base["C"].shape[0] - 4),
        "success": bool(res.success),
        "message": str(res.message),
        "elapsed_s": t1 - t0,
        "nfev": int(getattr(res, "nfev", 0)),
        "breakdown": {"chi2_sn": float(sn_fn(res.x)),
                      "chi2_bao": float(bao_fn(res.x))}
    }
    p = Path(args.out_prefix).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {p}")

if __name__ == "__main__":
    main()
