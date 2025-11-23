import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless backend
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from code.models.cosmology_basics import LCDM
from code.models.resonant_de import ResonantDE

MAN = "data_manifest.json"
with open(MAN) as f: man = json.load(f)
pant = [p for p in man["pantheon_plus"] if os.path.exists(p)]
dat  = sorted([p for p in pant if p.endswith(".dat")])[0]
cov  = sorted([p for p in pant if p.endswith(".cov")])[0]

df = pd.read_csv(dat, sep=r"\s+", engine="python", comment="#")
z  = df["zCMB"].to_numpy(float)
mu = df["MU_SH0ES"].to_numpy(float)
C  = np.loadtxt(cov); N = mu.size
if C.ndim==1 and C.size==N*N+1 and int(round(C[0]))==N: C = C[1:].reshape(N,N)
elif C.ndim==1 and C.size==N*N: C = C.reshape(N,N)

def profiled_residuals(mu_th):
    eps = 1e-9*np.median(np.diag(C)); Cf = C + np.eye(N)*eps
    cf  = cho_factor(Cf, lower=True, check_finite=False)
    Ci  = lambda v: cho_solve(cf, v, check_finite=False)
    ones = np.ones(N)
    r0 = mu - mu_th
    M  = float(ones @ Ci(r0) / (ones @ Ci(ones)))
    r  = r0 - M*ones
    return r, M

# Use your best fits printed earlier (adjust if your numbers differ)
lcdm = LCDM(H0=70.0, Omega_m=0.381)
r_l, M_l = profiled_residuals(lcdm.distance_modulus(z))

resn = ResonantDE(H0=70.0, Omega_m=0.338, A=0.20, f=5.0, phi=1.482)
r_r, M_r = profiled_residuals(resn.distance_modulus(z))

# Bin residuals in log(1+z) for clarity
x = np.log1p(z); order = np.argsort(x); xb = x[order]
def binplot(x, y, nb=30):
    edges = np.linspace(x.min(), x.max(), nb+1)
    xc = 0.5*(edges[1:]+edges[:-1]); ym=[]; ye=[]
    for lo,hi in zip(edges[:-1], edges[1:]):
        m = (x>=lo)&(x<hi)
        if m.sum()>0:
            ym.append(y[m].mean()); ye.append(y[m].std()/max(1,(m.sum())**0.5))
        else:
            ym.append(np.nan); ye.append(np.nan)
    return xc, np.array(ym), np.array(ye)

xc, yl, el = binplot(xb, r_l[order], nb=30)
_,  yr, er = binplot(xb, r_r[order], nb=30)

plt.figure(figsize=(8,4.5))
plt.axhline(0, ls="--")
plt.errorbar(xc, yl, el, fmt=".", label="ΛCDM residuals (binned)")
plt.errorbar(xc, yr, er, fmt=".", label="Resonant residuals (binned)")
plt.xlabel("log(1+z)")
plt.ylabel("μ_obs − μ_model − M  [mag]")
plt.legend()
plt.tight_layout()
plt.savefig("sn_residuals.png", dpi=160)
print("Saved: sn_residuals.png")
