#!/usr/bin/env python3
"""
Resonant-medium BAO scan:
- Reads BAO measurements (z, DM_over_r_d, DH_over_r_d) and covariance
- Fits LCDM baseline by profiling β = 1/r_d
- For each resonant frequency f_R, builds two linear templates (φ=0 and φ=π/2)
  and solves a 3-parameter GLS: [β, β*A_c, β*A_s]
- Reports: Δχ² (vs LCDM), AICc/BIC, best amplitude A = sqrt((A_c)^2+(A_s)^2),
  best phase, and a 95% CL upper limit on A with φ profiled (Δχ²=2.71)

Frequency convention: cycles per ln(1+z). Model (small-amplitude):
  H_res(z) = H_LCDM(z) * [1 + A * sin(2π f_R ln(1+z) + φ)]
Distances DM, DH recomputed with H_res (β scales 1/r_d as usual).

Usage:
  python3 tools/resonant_bao_scan.py \
    --bao-csv bao_measurements.csv --cov-csv bao_covariance.csv \
    --out outputs/res_scan --fmin 0.05 --fmax 2.00 --nf 60 \
    --Om0 0.3 --h 0.7 --stack-style block
"""

import argparse, json, math, os
import numpy as np, pandas as pd
from pathlib import Path

# ---------- numerics ----------
try:
    from scipy.linalg import cho_factor, cho_solve
    def inv_op(C):
        eps = 1e-12 * float(np.median(np.diag(C)))
        Cf  = C + eps*np.eye(C.shape[0])
        cf  = cho_factor(Cf, lower=True, check_finite=False)
        return lambda v: cho_solve(cf, v, check_finite=False)
except Exception:
    def inv_op(C):
        eps = 1e-12 * float(np.median(np.diag(C)))
        Cf  = C + eps*np.eye(C.shape[0])
        Ci  = np.linalg.inv(Cf)
        return lambda v: Ci @ v

C_KM_S = 299792.458

def E_LCDM(z, Om0): return np.sqrt(Om0*(1+z)**3 + (1-Om0))
def H_LCDM(z, h, Om0): return 100.0*h*E_LCDM(z, Om0)  # km/s/Mpc

def H_RES(z, h, Om0, A, f, phi):
    # multiplicative modulation of H with small A; keep positive
    arg = 2.0*np.pi*f*np.log1p(z) + phi
    mod = 1.0 + A*np.sin(arg)
    # safety clamp to avoid sign flip if user scans too large A
    mod = np.clip(mod, 0.5, 1.5)
    return H_LCDM(z, h, Om0) * mod

def DM_DH_from_H(z, H_of_z):
    # compute DM(z) = c * ∫ dz'/H(z'), and DH(z)=c/H(z)
    z = np.asarray(z, float)
    # integration by trapezoid on a fine grid per target z to be accurate & simple
    DM = np.zeros_like(z, float)
    for i, zi in enumerate(z):
        if zi <= 0: DM[i] = 0.0
        else:
            zz = np.linspace(0.0, zi, 2048)
            Hz = H_of_z(zz)
            DM[i] = C_KM_S * np.trapz(1.0/Hz, zz)
    DH = C_KM_S / H_of_z(z)
    return DM, DH

def build_model_vec(z, DM, DH, style="block"):
    if style == "block":
        return np.concatenate([DM, DH])
    elif style == "pair":
        out = []
        for i in range(len(z)):
            out.append(DM[i]); out.append(DH[i])
        return np.array(out, float)
    else:
        raise SystemExit("Unknown stack style: "+style)

def read_bao_table(path, stack_style="block"):
    df = pd.read_csv(path)
    # Expect columns like: z, DM_over_r_d, DH_over_r_d
    low = {c.lower(): c for c in df.columns}
    zc   = low.get("z")
    dmc  = low.get("dm_over_r_d") or low.get("dm/rd") or low.get("dmrd")
    dhc  = low.get("dh_over_r_d") or low.get("dh/rd") or low.get("dhrd")
    if not (zc and dmc and dhc):
        raise SystemExit(f"[bao] need columns z, DM_over_r_d, DH_over_r_d. Got {list(df.columns)}")
    z  = pd.to_numeric(df[zc],  errors="coerce").to_numpy()
    yD = pd.to_numeric(df[dmc], errors="coerce").to_numpy()
    yH = pd.to_numeric(df[dhc], errors="coerce").to_numpy()
    if stack_style == "block":
        y = np.concatenate([yD, yH])
    elif stack_style == "pair":
        y = np.empty(2*len(z), float); y[0::2]=yD; y[1::2]=yH
    else:
        raise SystemExit("Unknown stack style: "+stack_style)
    return z, y

def profile_beta(y, m, Ci):
    # GLS: β = (m^T C^-1 y) / (m^T C^-1 m); χ² = (y-βm)^T Ci (y-βm)
    num = float(m @ Ci(y))
    den = float(m @ Ci(m))
    den = den if np.isfinite(den) and den>1e-300 else 1e-300
    beta = num / den
    r = y - beta*m
    chi2 = float(r @ Ci(r))
    return beta, chi2

def gls_fit(y, cols, Ci):
    # GLS coefficients for design matrix with columns 'cols' (list of arrays)
    X = np.stack(cols, axis=1)  # shape N x p
    # Solve (X^T C^-1 X) b = X^T C^-1 y
    XT_Ci = np.stack([Ci(X[:,j]) for j in range(X.shape[1])], axis=1)  # N x p
    M  = X.T @ XT_Ci                      # p x p
    rhs= X.T @ Ci(y)                      # p
    # jitter
    eps = 1e-14 * float(np.median(np.diag(M))) if np.all(np.isfinite(M)) else 1e-14
    M  = M + eps*np.eye(M.shape[0])
    b  = np.linalg.solve(M, rhs)
    r  = y - X @ b
    chi2 = float(r @ Ci(r))
    return b, chi2, M

def aicc(chi2, k, n):
    if n - k - 1 <= 0:  # protect small-sample blowups
        return chi2 + 2*k
    return chi2 + 2*k + (2*k*(k+1))/(n - k - 1)

def bic(chi2, k, n):
    return chi2 + k*math.log(n)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bao-csv", required=True)
    ap.add_argument("--cov-csv", required=True)
    ap.add_argument("--out", default="outputs/res_scan")
    ap.add_argument("--Om0", type=float, default=0.3)
    ap.add_argument("--h",   type=float, default=0.7)
    ap.add_argument("--stack-style", choices=["block","pair"], default="block")
    ap.add_argument("--fmin", type=float, default=0.05)
    ap.add_argument("--fmax", type=float, default=2.00)
    ap.add_argument("--nf",   type=int,   default=60)
    ap.add_argument("--deltaA", type=float, default=0.01, help="finite-diff amplitude for templates")
    ap.add_argument("--amax_ul", type=float, default=0.5, help="max A scanned for 95% limit")
    args = ap.parse_args()

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    # Data
    z, y = read_bao_table(args.bao_csv, args.stack_style)
    C = np.loadtxt(args.cov_csv, delimiter=",").astype(float)
    if C.shape != (len(y), len(y)): raise SystemExit(f"[cov] shape {C.shape} != {(len(y), len(y))}")
    Ci = inv_op(C)
    n  = len(y)

    # Baseline ΛCDM model vector
    DM0, DH0 = DM_DH_from_H(z, lambda zz: H_LCDM(zz, args.h, args.Om0))
    m0 = build_model_vec(z, DM0, DH0, args.stack_style)
    beta0, chi2_0 = profile_beta(y, m0, Ci)
    aic0 = aicc(chi2_0, k=1, n=n)
    bic0 = bic (chi2_0, k=1, n=n)

    print(f"[LCDM] beta={beta0:.8f}  r_d={1.0/beta0:.2f} Mpc  chi2={chi2_0:.3f}  dof≈{n-1}  AICc={aic0:.3f}  BIC={bic0:.3f}")

    # Frequency grid
    freqs = np.linspace(args.fmin, args.fmax, args.nf)
    rows  = []

    for f in freqs:
        dA = args.deltaA
        # Templates at φ=0 and φ=π/2, finite diff around A=0
        def m_res(A, phi):
            DM, DH = DM_DH_from_H(z, lambda zz: H_RES(zz, args.h, args.Om0, A, f, phi))
            return build_model_vec(z, DM, DH, args.stack_style)

        t_c = (m_res(dA, 0.0)         - m0)/dA
        t_s = (m_res(dA, 0.5*np.pi)   - m0)/dA

        # GLS with columns [m0, t_c, t_s] → coefficients [β, β*A_c, β*A_s]
        b, chi2, M = gls_fit(y, [m0, t_c, t_s], Ci)
        beta, b_c, b_s = float(b[0]), float(b[1]), float(b[2])
        A_c, A_s = (b_c/beta), (b_s/beta)
        A_mag = float(np.hypot(A_c, A_s))
        phi   = float(math.atan2(A_s, A_c))  # best phase
        aic   = aicc(chi2, k=3, n=n)
        bicv  = bic (chi2, k=3, n=n)
        dchi  = chi2_0 - chi2
        daic  = aic0   - aic
        dbic  = bic0   - bicv

        # --- 95% CL upper limit on A with φ profiled ---
        # move along best phase direction t_eff = cosφ t_c + sinφ t_s
        t_eff = math.cos(phi)*t_c + math.sin(phi)*t_s

        def chi2_at_ampl(A):
            # model column becomes m0 + A * t_eff; β refit (1-param GLS)
            mA = m0 + A * t_eff
            _, chi2A = profile_beta(y, mA, Ci)
            return chi2A

        # bracket Δχ²=2.71
        chi2_min = chi2
        target = chi2_min + 2.71
        lo, hi = 0.0, args.amax_ul
        chi_lo, chi_hi = chi2_at_ampl(lo), chi2_at_ampl(hi)
        A95 = np.nan
        if chi_hi >= target:
            # bisection
            for _ in range(40):
                mid = 0.5*(lo+hi)
                chim = chi2_at_ampl(mid)
                if chim >= target:
                    hi, chi_hi = mid, chim
                else:
                    lo, chi_lo = mid, chim
            A95 = 0.5*(lo+hi)

        rows.append(dict(f=f, beta=beta, A_best=A_mag, phi_best=phi,
                         chi2=chi2, dchi_improve=dchi,
                         AICc=aic, dAICc=daic, BIC=bicv, dBIC=dbic,
                         A95=A95))

    out = pd.DataFrame(rows).sort_values("f").reset_index(drop=True)
    out_csv = args.out + "_periodogram.csv"
    out.to_csv(out_csv, index=False)
    # print top few by Δχ²
    tops = out.sort_values("dchi_improve", ascending=False).head(10)
    print("\nTop-by Δχ² improvement (vs LCDM):")
    print(tops.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # save a tiny JSON summary
    js = {
        "lcdm": {"beta": beta0, "r_d": 1.0/beta0, "chi2": chi2_0, "AICc": aic0, "BIC": bic0, "n": n},
        "scan": {"fmin": args.fmin, "fmax": args.fmax, "nf": args.nf,
                 "deltaA": args.deltaA, "periodogram_csv": out_csv}
    }
    with open(args.out + "_summary.json","w") as f:
        json.dump(js, f, indent=2)
    print(f"\nWrote:\n  {out_csv}\n  {args.out}_summary.json")
    print("\nColumns: f, A_best, phi_best, chi2, dchi_improve, AICc, dAICc, BIC, dBIC, A95")
if __name__ == "__main__":
    main()
