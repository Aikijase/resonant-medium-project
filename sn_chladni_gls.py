#!/usr/bin/env python3
import argparse, json, math, numpy as np, pandas as pd
from numpy.linalg import cholesky, solve
import matplotlib.pyplot as plt

def load_cov(path, N_hint=None):
    txt = open(path,'r').read().strip()
    try:
        arr = json.loads(txt); C = np.array(arr, dtype=float)
    except Exception:
        try: C = np.loadtxt(path, delimiter=",")
        except Exception: C = np.loadtxt(path)
    if C.ndim==1:
        nflat=C.size; n=int(round((nflat)**0.5))
        if N_hint is not None: n = int(N_hint)
        C=C[:n*n].reshape(n,n)
    return C

def nearest_spd(C):
    B = (C + C.T)/2.0
    U, s, Vt = np.linalg.svd(B)
    H = (Vt.T) @ np.diag(np.maximum(s, 1e-12)) @ Vt
    Csp = (B + H)/2.0
    Csp = (Csp + Csp.T)/2.0
    jitter = 0.0
    for _ in range(8):
        try:
            cholesky(Csp + jitter*np.eye(Csp.shape[0])); return Csp + jitter*np.eye(Csp.shape[0])
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter==0 else jitter*10
    w, V = np.linalg.eigh(Csp); w = np.maximum(w, 1e-12)
    return (V @ np.diag(w) @ V.T)

def normalize(s):  return "".join(ch.lower() for ch in s if ch.isalnum())
def find_sn_cols(df, zpref="auto", zcol_override=None, mucol_override=None):
    if zcol_override and mucol_override: return zcol_override, mucol_override
    norm = {normalize(c): c for c in df.columns}
    z_groups = {
        "zhd":  ["zhd","z_hd","zheliohd","zhelhd","hdz"],
        "zcmb": ["zcmb","z_cmb","zcmbhd","zcmbhel","cmbz"],
        "zhel": ["zhel","zheliocentric","zhelio"],
        "z":    ["z"],
    }
    mu_cands = ["mush0es","mushoes","mu","mucorr","mucorrected","mubest","distmod","dlmag","dmod","distancemodulus"]
    def pick_z(group):
        for key in z_groups[group]:
            if key in norm: return norm[key]
        return None
    zcol = zcol_override
    if zcol is None:
        if zpref in ("zhd","zcmb","zhel","z"): zcol = pick_z(zpref)
        if zcol is None:
            for g in ("zhd","zcmb","zhel","z"):
                zcol = pick_z(g)
                if zcol: break
    mucol = mucol_override
    if mucol is None:
        for key in mu_cands:
            if key in norm: mucol = norm[key]; break
        if mucol is None:
            for c in df.columns:
                nc = normalize(c)
                if ("mu" in nc or "distmod" in nc or "dlmag" in nc) and not ("err" in nc or "sigma" in nc or "cov" in nc):
                    mucol = c; break
    if zcol is None or mucol is None:
        raise SystemExit(f"Could not detect z/mu columns.\nHave: {list(df.columns)[:20]}\nDetected z={zcol} mu={mucol}")
    return zcol, mucol

def r_of_z(z, a,b, O0,O1):
    Om = O0 + O1*(z/(1.0+z))
    k  = a*Om + b
    s  = 1.0/np.where(k>0, k, np.nan)
    return s/s[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--cov", default=None, help="full covariance (required unless --use_diag)")
    ap.add_argument("--a", type=float, default=0.078)
    ap.add_argument("--b", type=float, default=-0.011)
    ap.add_argument("--out_prefix", default="sn_fit")
    ap.add_argument("--grid_O0", nargs=3, type=float, default=[1.0, 2.4, 0.05])
    ap.add_argument("--grid_O1", nargs=3, type=float, default=[-0.6, 1.4, 0.05])
    ap.add_argument("--zpref", choices=["auto","zcmb","zhd","zhel","z"], default="auto")
    ap.add_argument("--zcol", default=None); ap.add_argument("--mucol", default=None)
    ap.add_argument("--use_diag", action="store_true", help="use diagonal errors from table")
    ap.add_argument("--errcol", default="MU_SH0ES_ERR_DIAG", help="column for per-SN σ when --use_diag")
    ap.add_argument("--sigma_int", type=float, default=0.12, help="add intrinsic scatter in mag (diag mode)")
    ap.add_argument("--maxN", type=int, default=None)
    ap.add_argument("--progress", action="store_true")
    args=ap.parse_args()

    # robust CSV read
    df = pd.read_csv(args.table, sep=None, engine="python", comment="#")
    if (len(df.columns)==1) or all(str(c).replace('.','',1).isdigit() for c in df.columns):
        df = pd.read_csv(args.table, sep=None, engine="python", comment="#", header=None)
        if df.shape[0]>1 and not all(str(x).replace('.','',1).isdigit() for x in df.iloc[0].values):
            df.columns = [str(x) for x in df.iloc[0]]; df = df.iloc[1:].reset_index(drop=True)

    zcol, mucol = find_sn_cols(df, zpref=args.zpref, zcol_override=args.zcol, mucol_override=args.mucol)
    z  = df[zcol].to_numpy(float)
    mu = df[mucol].to_numpy(float)

    if args.use_diag:
        if args.errcol not in df.columns:
            raise SystemExit(f"Error column '{args.errcol}' not found. Have: {list(df.columns)[:30]}")
        sig = df[args.errcol].to_numpy(float)
        L   = len(mu); z, mu, sig = z[:L], mu[:L], sig[:L]
        C   = np.diag(np.maximum(sig, 1e-6)**2)
    else:
        if args.cov is None:
            raise SystemExit("Provide --cov, or use --use_diag to ignore the full covariance.")
        C   = load_cov(args.cov, N_hint=len(mu))
        L   = min(len(mu), C.shape[0]); z, mu, C = z[:L], mu[:L], C[:L,:L]

    if args.maxN is not None:
        L = min(len(mu), args.maxN); z, mu, C = z[:L], mu[:L], C[:L,:L]

    # pre-factorize once
    Lfac = cholesky(nearest_spd(C))
    y    = solve(Lfac, mu)

    O0s = np.arange(*args.grid_O0); O1s = np.arange(*args.grid_O1)
    total = len(O0s)*len(O1s); best = {"chi2":1e99}; n=0

    for O0 in O0s:
        for O1 in O1s:
            r = r_of_z(z, args.a, args.b, O0, O1)
            x = np.log10(r + 1e-300)
            X = np.column_stack([np.ones_like(x), x])
            Xp = solve(Lfac, X)
            theta, *_ = np.linalg.lstsq(Xp, y, rcond=None)
            resid = y - Xp @ theta
            chi2 = float(resid @ resid)
            if chi2 < best["chi2"]:
                best = dict(O0=float(O0), O1=float(O1),
                            M=float(theta[0]), beta=float(theta[1]),
                            chi2=chi2, dof=int(len(mu)-2),
                            zcol=zcol, mucol=mucol)
            n += 1
            if args.progress and (n % 200 == 0):
                print(f"[{n}/{total}] best χ²={best['chi2']:.2f} at O0={best['O0']:.2f}, O1={best['O1']:.2f}")

    r  = r_of_z(z, args.a, args.b, best["O0"], best["O1"])
    x  = np.log10(r + 1e-300); M, beta = best["M"], best["beta"]
    mu_model = M + beta*x

    pd.DataFrame({"z":z, "mu":mu, "mu_model":mu_model, "log10_r":x}).to_csv(f"{args.out_prefix}_overlay.csv", index=False)
    plt.figure(figsize=(7,4.2), dpi=140)
    plt.plot(z, mu-mu_model, 'o', ms=2); plt.axhline(0, lw=1)
    plt.xlabel("z"); plt.ylabel("μ − μ_model")
    plt.title(f"Pantheon+SH0ES vs Chladni-relative | χ²/dof = {best['chi2']/max(best['dof'],1):.2f}")
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_residuals.png"); plt.close()

    with open(f"{args.out_prefix}_summary.json","w") as f: json.dump(best, f, indent=2)
    print(json.dumps(best, indent=2))

if __name__=="__main__": main()
