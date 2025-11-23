import json, numpy as np, pandas as pd, os, math
from math import sqrt
from scipy.integrate import solve_ivp

def H_LCDM(z, H0=70.0, Om=0.3, Ol=0.7):
    return H0 * sqrt(Om*(1+z)**3 + Ol)

def dlnH_dlnA_LCDM(z, Om=0.3, Ol=0.7, dz=1e-4):
    H1 = H_LCDM(z+dz, Om=Om, Ol=Ol)
    H2 = H_LCDM(z-dz, Om=Om, Ol=Ol)
    dlnH_dz = (np.log(H1) - np.log(H2))/(2*dz)
    return -(1+z)*dlnH_dz

def solve_growth_LCDM(zmax=2.0, npts=200, sigma8_0=0.8, Om=0.3, Ol=0.7):
    ln_a = np.linspace(np.log(1/(1+zmax)), 0.0, npts)  # ascending
    def ode(ln_a, y):
        a = np.exp(ln_a); z = 1/a - 1
        D, Dp = y
        dlnH = dlnH_dlnA_LCDM(z, Om=Om, Ol=Ol)
        # Simple GR-like schematic form; same as your resonant solver for fair compare
        Dpp = -(2.0 + dlnH)*Dp + 1.5*D
        return [Dp, Dpp]
    sol = solve_ivp(ode, (ln_a[0], ln_a[-1]), [1.0, 0.0], t_eval=ln_a, rtol=1e-6, atol=1e-8)
    a = np.exp(ln_a); z = 1/a - 1
    D = sol.y[0]; D /= max(D[-1], 1e-12)
    f_log = np.gradient(np.log(np.clip(D,1e-12,None)), ln_a, edge_order=2)
    fs8 = sigma8_0 * f_log * D
    return z, fs8

if __name__ == "__main__":
    z, fs8 = solve_growth_LCDM()
    os.makedirs("outputs/phase2", exist_ok=True)
    pd.DataFrame({"z": z, "fs8_pred": fs8}).to_csv("outputs/phase2/fs8_eval_lcdm.csv", index=False)
    # Score vs the same data file your runner uses so AIC/BIC are comparable
    import pandas as pd
    D = pd.read_csv("data/growth/fs8_catalog.csv")
    fs8_at_data = np.interp(D["z"].values, z, fs8)
    resid = (fs8_at_data - D["fs8_obs"].values)/D["sigma"].values
    chi2 = float((resid**2).sum()); n = int(len(D)); k = 0
    AIC = chi2 + 2*k; BIC = chi2 + k*math.log(max(n,1))
    ref = {
        "probe": "fs8", "n": n, "k": k, "chi2": chi2,
        "AIC": {"value": AIC, "delta": 0.0},
        "BIC": {"value": BIC, "delta": 0.0},
        "WWI": 0.0
    }
    json.dump(ref, open("outputs/phase2/fs8_eval_lcdm.json","w"), indent=2)
    print("Wrote outputs/phase2/fs8_eval_lcdm.{json,csv}")
