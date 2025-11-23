import math, json, sys, os

def load_j(path):
    with open(path) as f: return json.load(f)

def aic_bic(J):
    chi2 = float(J["chi2"])
    k    = len(J.get("best_params", {}))
    N    = int(J.get("ndof", 0)) + k if "ndof" in J else int(J.get("N", 0) or 1700)  # fallback
    AIC  = chi2 + 2*k
    BIC  = chi2 + k*math.log(max(N, 1))
    return chi2, k, N, AIC, BIC

def wwi(delta_aic, delta_bic):
    # Watts Win Index (0–100)
    vA = max(0.0, min(1.0, (-delta_aic)/10.0))
    vB = max(0.0, min(1.0, (-delta_bic)/10.0))
    return 100.0*min(vA, vB)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        J = load_j(sys.argv[1])
        chi2,k,N,AIC,BIC = aic_bic(J)
        print(f"file={sys.argv[1]}")
        print(f"χ²={chi2:.3f}  k={k}  N={N}")
        print(f"AIC={AIC:.3f}  BIC={BIC:.3f}")
    elif len(sys.argv) == 3:
        J0 = load_j(sys.argv[1]); J1 = load_j(sys.argv[2])
        c0,k0,N0,A0,B0 = aic_bic(J0); c1,k1,N1,A1,B1 = aic_bic(J1)
        dAIC = A1 - A0; dBIC = B1 - B0; dchi2 = c1 - c0
        print(f"baseline={os.path.basename(sys.argv[1])}  cand={os.path.basename(sys.argv[2])}")
        print(f"Δχ²={dchi2:.3f}  ΔAIC={dAIC:.3f}  ΔBIC={dBIC:.3f}")
        print(f"WWI={wwi(dAIC,dBIC):.1f}/100")
    else:
        print("usage:\n  python3 tools/metrics.py <result.json>\n  python3 tools/metrics.py <baseline.json> <candidate.json>")
        sys.exit(2)
