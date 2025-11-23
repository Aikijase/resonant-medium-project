#!/usr/bin/env python3
from pathlib import Path
import sys, json, math, argparse, time
import numpy as np
import pandas as pd

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_fit_resonant_rd as J  # uses: make_sn_chi2, make_bao_chi2_both

def parse_args():
    p = argparse.ArgumentParser(description="Resonant BAO+SN fitter (self-contained loaders).")
    p.add_argument("--bao-csv", required=True)
    p.add_argument("--bao-cov", required=True)
    p.add_argument("--sn-csv",  required=True)
    p.add_argument("--sn-cov",  required=True)
    p.add_argument("--out-prefix", required=True)

    p.add_argument("--assume-per-rd", action="store_true",
                   help="Treat DESI vector as y=/r_d basis (expected for your DR1 long vector).")
    p.add_argument("--plus-lya", action="store_true",
                   help="Keep Lyα rows if present in the covariance.")
    p.add_argument("--diag-only", action="store_true",
                   help="Ignore off-diagonals (smoke test).")

    p.add_argument("--H0", type=float, default=70.0)
    p.add_argument("--Om", type=float, default=0.3)
    p.add_argument("--rd", type=float, default=147.1)

    p.add_argument("--max-evals", type=int, default=250)
    p.add_argument("--optimizer", default="lbfgsb", choices=["lbfgsb","nelder","powell"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--A0", type=float, default=0.4)
    p.add_argument("--f0", type=float, default=2.4)
    p.add_argument("--phi0", type=float, default=1.1)
    p.add_argument("--gamma0", type=float, default=0.0)
    return p.parse_args()

# ---------- loaders (self-contained) ----------
def _load_cov(csv_path: str, diag_only: bool):
    C = np.asarray(pd.read_csv(csv_path, header=None), dtype=float)
    if diag_only:
        C = np.diag(np.diag(C))
    return C

def _chol_or_prec(M: np.ndarray):
    """Return (L_or_R, is_prec). If M is SPD covariance, return (L, False) with L s.t. L L^T = C.
       If it looks like a precision matrix (big diagonals), return (R, True) with R s.t. R^T R = P."""
    try:
        L = np.linalg.cholesky(M)
        return L, False
    except np.linalg.LinAlgError:
        # Try treating as precision
        try:
            R = np.linalg.cholesky(M)
            return R, True
        except Exception as e:
            # Last resort: nudge the diagonal
            eps = 1e-8
            L = np.linalg.cholesky(M + eps*np.eye(M.shape[0]))
            return L, False

def load_sn_pack(sn_csv: str, sn_cov_csv: str, diag_only: bool):
    # Expect columns: z, mu (and maybe sigma_mu, which we ignore since full cov is supplied)
    df = pd.read_csv(sn_csv)
    z  = np.asarray(df["z"], dtype=float)
    mu = np.asarray(df["mu"], dtype=float)

    C = _load_cov(sn_cov_csv, diag_only=diag_only)   # SPD covariance
    if C.shape[0] != mu.shape[0]:
        raise ValueError(f"SN cov shape {C.shape} != SN vector length {mu.shape[0]}")
    L_or_R, is_prec = _chol_or_prec(C)
    return {"z": z, "y": mu, "LR": L_or_R, "is_prec": is_prec}

def load_bao_pack(bao_csv: str, bao_cov_csv: str, diag_only: bool):
    # Expect long vector with columns that include the observable 'y' and optionally 'sigma'.
    df = pd.read_csv(bao_csv)
    # Use the provided long vector ordering as-is:
    # try common column names; fallback to the last numeric column.
    for col in ["y","obs","value","meas","x"]:
        if col in df.columns:
            y = np.asarray(df[col], dtype=float); break
    else:
        # last numeric column
        y = np.asarray(df.select_dtypes(include=[np.number]).iloc[:,-1], dtype=float)

    C = _load_cov(bao_cov_csv, diag_only=diag_only)
    if C.shape[0] != y.shape[0]:
        raise ValueError(f"BAO cov shape {C.shape} != BAO vector length {y.shape[0]}")
    # Some pipelines also carry per-point sigma. Keep if present (not required if full C given).
    sigma = df["sigma"].to_numpy(dtype=float) if "sigma" in df.columns else None
    return {"y": y, "sigma": sigma, "C": C}

# ---------- objective & optimizer ----------
import inspect

def _call_make_sn_chi2(J, sn_pack, H0, Om):
    fn = J.make_sn_chi2
    params = list(inspect.signature(fn).parameters.keys())
    # Try common variants:
    if params[:4] == ["sn_pack","is_prec","H0","Om"] or "is_prec" in params and "H0" in params and "Om" in params:
        return fn(sn_pack, sn_pack["is_prec"], H0, Om)
    if params[:4] == ["sn_pack","is_prec","Om","H0"]:
        return fn(sn_pack, sn_pack["is_prec"], Om, H0)
    if params[:3] == ["sn_pack","H0","Om"]:
        return fn(sn_pack, H0, Om)
    if params[:3] == ["sn_pack","Om","H0"]:
        return fn(sn_pack, Om, H0)
    raise RuntimeError(f"Unsupported make_sn_chi2 signature: {params}")

def make_objective(args):
    sn_pack  = load_sn_pack(args.sn_csv, args.sn_cov, diag_only=args.diag_only)
    bao_pack = load_bao_pack(args.bao_csv, args.bao_cov, diag_only=args.diag_only)

    chi2_sn  = J.make_sn_chi2(sn_pack, sn_pack["is_prec"], args.H0, args.Om)

    chi2_bao = J.make_bao_chi2_both(bao_pack, args.H0, args.Om, args.rd, # expects dict with y/C (and sigma optional)
                                    fit_Om=False, fit_rd=False, nz_bg=800)

    def chi2(theta):
        return float(chi2_sn(theta) + chi2_bao(theta))
    return chi2, sn_pack, bao_pack

def minimize(chi2, x0, method, max_evals, seed):
    import scipy.optimize as opt
    rng = np.random.default_rng(seed)
    if method == "lbfgsb":
        bounds = [(0.0, 5.0), (0.0, 8.0), (-math.pi, math.pi), (0.0, 2.0)]
        res = opt.minimize(chi2, x0, method="L-BFGS-B",
                           bounds=bounds, options={"maxfun": max_evals})
    elif method == "nelder":
        res = opt.minimize(chi2, x0, method="Nelder-Mead",
                           options={"maxfev": max_evals})
    else:
        res = opt.minimize(chi2, x0, method="Powell",
                           options={"maxfev": max_evals})
    return res

def main():
    args = parse_args()
    chi2, sn_pack, bao_pack = make_objective(args)

    x0 = np.array([args.A0, args.f0, args.phi0, args.gamma0], dtype=float)
    t0 = time.time()
    res = minimize(chi2, x0, args.optimizer, args.max_evals, args.seed)
    t1 = time.time()

    A,f,phi,gamma = [float(v) for v in res.x]
    chi2_min = float(res.fun)

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
