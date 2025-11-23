#!/usr/bin/env python3
import argparse, json, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def load_cov(path):
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception:
        return np.loadtxt(path)

def pick_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low: return low[n]
    return None

def dv_and_cov_rel(z, DM, DH, C_block):
    """Return (r_full, C_r, mask_nonref).
    r_full has length N with r[0]=1; C_r is (N-1)x(N-1) for ratios i>0."""
    N = len(z)
    DV = (DM**2 * (z*DH))**(1.0/3.0)         # DV or DV/rd (relative works the same)
    # Jacobian: DV wrt [DM1..DMN, DH1..DHN]
    dDV_dDM = (1.0/3.0)*DV*(2.0/DM)          # length N
    dDV_dDH = (1.0/3.0)*DV*(1.0/DH)          # length N
    J = np.zeros((N, 2*N))
    for i in range(N):
        J[i, i]     = dDV_dDM[i]
        J[i, N + i] = dDV_dDH[i]
    C_DV = J @ C_block @ J.T

    # Build linear map A so that r_i (i>0) = DV_i/DV_0 = A_i * DV
    A = np.zeros((N-1, N))
    DV0 = DV[0]
    for i in range(1, N):
        A[i-1, i] = 1.0 / DV0
        A[i-1, 0] = - DV[i] / (DV0**2)
    C_r = A @ C_DV @ A.T

    r_full = np.ones(N)
    r_full[1:] = DV[1:] / DV0
    mask = np.ones(N, dtype=bool); mask[0] = False
    return r_full, C_r, mask

def chladni_rel(z, a, b, O0, O1, law):
    if law == "const":
        Om = O0 + 0.0*z
    elif law == "z_over_1pz":
        Om = O0 + O1*(z/(1.0+z))
    elif law == "ln1pz":
        Om = O0 + O1*np.log(1.0+z)
    elif law == "sqrt1pz":
        Om = O0 + O1*(np.sqrt(1.0+z)-1.0)
    else:
        raise ValueError("unknown law")
    k = a*Om + b
    s = 1.0/np.where(k>0, k, np.nan)
    return s/s[0]

def Om_piecewise(z, ctrl_z, ctrl_Om):
    return np.interp(z, ctrl_z, ctrl_Om)

def _fit_A_chi2(y, C, m):
    from numpy.linalg import cholesky, solve
    L = cholesky((C+C.T)/2.0 + 1e-12*np.eye(C.shape[0]))
    y2 = solve(L, y); m2 = solve(L, m)
    A  = float(m2 @ y2 / max(m2 @ m2, 1e-30))
    r  = y2 - A*m2
    return A, float(r @ r)

def chi2_rel(r_vec, C_sub, r_model):
    # small jitter for numerical stability
    Csym = (C_sub + C_sub.T)/2.0
    Csym.flat[::Csym.shape[0]+1] += 1e-12
    try:
        Ci = np.linalg.inv(Csym)
    except np.linalg.LinAlgError:
        Ci = np.linalg.pinv(Csym, rcond=1e-12)
    d = r_vec - r_model
    return float(d @ Ci @ d)

def permute_DM_DH(C):
    """Swap [DM... DH...] <-> [DH... DM...] blocks (2N x 2N)."""
    n2 = C.shape[0]; assert n2 % 2 == 0
    N  = n2 // 2
    P = np.zeros((n2, n2))
    for i in range(N):
        P[i, N+i]   = 1.0
        P[N+i, i]   = 1.0
    return P @ C @ P.T

def fit_all_laws(z, DM, DH, C_block, a, b):
    laws = ["const","z_over_1pz","ln1pz","sqrt1pz"]
    candidates = [("DM_DH", C_block), ("DH_DM", permute_DM_DH(C_block))]
    best_pack = None

    for tag, Cuse in candidates:
        r_full, C_r, mask = dv_and_cov_rel(z, DM, DH, Cuse)
        r_vec = r_full[mask]                 # length npts = N-1
        results = []; overlays = {}
        for law in laws:
            if law == "const":
                grid = [(O0, 0.0) for O0 in np.linspace(1.0, 2.4, 71)]
                kpar = 1
            else:
                O0s = np.linspace(1.0, 2.4, 36)
                O1s = np.linspace(-0.6, 1.4, 51)
                grid = [(O0, O1) for O0 in O0s for O1 in O1s]
                kpar = 2
            best = {"chi2": 1e99}
            for O0, O1 in grid:
                r_model_full = chladni_rel(z, a, b, O0, O1, law)
                chi2 = chi2_rel(r_vec, C_r, r_model_full[mask])
                if math.isfinite(chi2) and chi2 < best["chi2"]:
                    best.update(dict(O0=float(O0), O1=float(O1), chi2=float(chi2)))
            npts = len(r_vec)
            AIC  = best["chi2"] + 2*kpar
            AICc = AIC + (2*kpar*(kpar+1))/max(npts-kpar-1,1)
            BIC  = best["chi2"] + kpar*math.log(npts)
            best.update(dict(law=law, k=kpar, dof=npts-kpar, AIC=AIC, AICc=AICc, BIC=BIC))
            results.append(best)
            overlays[law] = chladni_rel(z, a, b, best["O0"], best["O1"], law)

        res = pd.DataFrame(results).sort_values("AICc")
        # remember the as-used r and covariance for overlay/sigmas
        if (best_pack is None) or (res.iloc[0]["chi2"] < best_pack[1].iloc[0]["chi2"]):
            best_pack = (tag, res, overlays, r_full, C_r, mask)

    return best_pack  # (order_tag, res_df, overlays, r_full, C_r, mask)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bao", required=True)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--a", type=float, default=0.078)
    ap.add_argument("--b", type=float, default=-0.011)
    ap.add_argument("--out_prefix", default="bao_fit_plus")
    args = ap.parse_args()

    df = pd.read_csv(args.bao)
    zcol = pick_col(df, ["z","zeff","z_eff"])
    DMcol = pick_col(df, ["dm_over_rd","d_m_over_rd","dm_over_r_d","d_m_over_r_d","dm","d_m"])
    DHcol = pick_col(df, ["dh_over_rd","d_h_over_rd","dh_over_r_d","d_h_over_r_d","dh","d_h"])
    if zcol is None or DMcol is None or DHcol is None:
        raise SystemExit(f"Missing columns. Have: {df.columns.tolist()}")

    z  = df[zcol].to_numpy(float)
    DM = df[DMcol].to_numpy(float)
    DH = df[DHcol].to_numpy(float)

    C = load_cov(args.cov)
    # Basic shape check
    N = len(z)
    if C.shape != (2*N, 2*N):
        raise SystemExit(f"Covariance shape {C.shape} != (2N,2N) with N={N}")

    order_tag, res, overlays, r_full, C_r, mask = fit_all_laws(z, DM, DH, C, args.a, args.b)

    # choose best law
    star = res.iloc[0]
    law_star = star["law"]
    r_model_full = overlays[law_star]

    # overlay CSV (include sigma; set sigma=0 for ref bin)
    sig = np.zeros_like(r_full)
    sig_sub = np.sqrt(np.clip(np.diag(C_r), 0, np.inf))
    sig[mask] = sig_sub
    ov = pd.DataFrame({"z": z, "DV_rel": r_full, "DV_rel_sigma": sig, "pred_rel_spacing": r_model_full})
    ov_path = f"{args.out_prefix}_overlay.csv"
    ov.to_csv(ov_path, index=False)

    # results CSV
    res_path = f"{args.out_prefix}_results.csv"
    res.to_csv(res_path, index=False)

    # plot
    plt.figure(figsize=(7,4.5), dpi=140)
    plt.errorbar(z, r_full, yerr=sig, fmt='o', capsize=3, label=r'BAO $D_V/r_d$ (relative)')
    plt.plot(z, r_model_full, marker='s',
             label=f'{law_star} ({order_tag}) | χ²/dof={(star["chi2"]/max(star["dof"],1)):.2f}, AICc={star["AICc"]:.1f}')
    plt.xlabel("Redshift z"); plt.ylabel("Relative scale (norm to first bin)")
    plt.legend(); plt.tight_layout()
    plot_path = f"{args.out_prefix}_plot.png"
    plt.savefig(plot_path); plt.close()

    print(json.dumps({"order": order_tag, "best": law_star,
                      "saved":[res_path, ov_path, plot_path]}, indent=2))

if __name__ == "__main__":
    main()
