import json, numpy as np, pathlib
from dataclasses import dataclass

# ----- Errors -----
class InputNotFound(RuntimeError):
    pass

# ----- File helpers -----
def _try_paths(cands):
    for p in cands:
        p = pathlib.Path(p)
        if p.exists():
            return p
    raise InputNotFound("Could not find residual series. Tried:\n  - " + "\n  - ".join(map(str, cands)))

def load_residual_series(user_path=None):
    """
    Returns (x, r) where x is the e-fold (or similar) grid and r are residuals.
    Accepted formats (first found wins):
      - CSV: columns x,res  (also accepts efold/k and residual/R/Pk_res aliases)
      - NPZ: arrays (x,res) or (k,Pk_res) or (efold,res)
      - JSON: {"x":[...], "res":[...]} or keys {"k","Pk_res"} or {"x","R"}
    """
    candidates = [user_path] if user_path else []
    candidates += [
        "outputs/phase8/residuals.csv",
        "outputs/phase8/residuals.json",
        "outputs/phase8/spectrum.npz",
        "outputs/phase8/residual_series.npz"
    ]
    p = _try_paths([c for c in candidates if c])

    if p.suffix.lower() == ".csv":
        import csv
        xs, rs = [], []
        with open(p) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                x = row.get("x", row.get("efold", row.get("k")))
                r = row.get("res", row.get("residual", row.get("R", row.get("Pk_res"))))
                if x is None or r is None:
                    continue
                xs.append(float(x)); rs.append(float(r))
        if not xs:
            raise InputNotFound(f"{p} had no usable x,res columns")
        return np.asarray(xs), np.asarray(rs)

    if p.suffix.lower() == ".json":
        J = json.load(open(p))
        if "x" in J and ("res" in J or "R" in J):
            return np.asarray(J["x"]), np.asarray(J.get("res", J.get("R")))
        if "k" in J and "Pk_res" in J:
            return np.asarray(J["k"]), np.asarray(J["Pk_res"])
        raise InputNotFound(f"{p} JSON lacked expected keys")

    if p.suffix.lower() == ".npz":
        Z = np.load(p)
        for xk, rk in (("x","res"), ("k","Pk_res"), ("efold","res")):
            if xk in Z.files and rk in Z.files:
                return np.asarray(Z[xk]), np.asarray(Z[rk])
        raise InputNotFound(f"{p} NPZ lacked expected arrays")

    raise InputNotFound(f"Unsupported residual format: {p}")

# ----- DSP helpers -----
def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p

def apply_window(y, kind="rect"):
    kind = (kind or "rect").lower()
    if kind in ("rect", "boxcar", "none"):
        return y
    if kind == "hann":
        return y * np.hanning(y.size)
    if kind == "blackman":
        return y * np.blackman(y.size)
    raise ValueError(f"Unknown window: {kind}")

@dataclass
class LorentzFit:
    A: float
    x0: float
    gamma: float
    chi2: float

def fit_lorentz(x, y):
    """
    Robust Lorentzian fit to (x, |y|). Returns A/(1+((x-x0)/gamma)^2).
    """
    from scipy.optimize import least_squares
    x = np.asarray(x); y = np.abs(np.asarray(y))
    if y.size == 0 or np.nanmax(y) <= 0:
        return LorentzFit(0.0, float(x[0] if x.size else 0.0), 1.0, float("inf"))

    x0_seed = float(x[np.nanargmax(y)])
    A_seed  = float(np.nanmax(y))
    g_seed  = (float(x.max()) - float(x.min())) / 20.0 if x.size > 1 else 1.0

    def model(p):
        A, x0, g = p
        return A / (1.0 + ((x - x0) / g) ** 2)

    def resid(p):
        return model(p) - y

    p0 = np.array([A_seed, x0_seed, max(g_seed, 1e-6)])
    lo = np.array([0.0, float(x.min()), max((x.max()-x.min())/200.0, 1e-6)])
    hi = np.array([10*A_seed + 1e-12, float(x.max()), max((x.max()-x.min())/2.0, 1e-3)])
    sol = least_squares(resid, p0, bounds=(lo, hi), verbose=0)
    r = resid(sol.x)
    chi2 = float(np.dot(r, r))
    A, x0, g = map(float, sol.x)
    return LorentzFit(A, x0, g, chi2)

def periodogram(x, r, pad_mul=1, window="rect"):
    """
    Simple (windowed) periodogram of residuals r sampled on grid x (assumed uniform spacing).
    Returns (freqs, power).
    """
    x = np.asarray(x); r = np.asarray(r)
    if x.size != r.size:
        raise ValueError(f"x and r must have same length, got {x.size} vs {r.size}")
    if x.size < 4:
        raise ValueError("Need at least 4 samples for a meaningful periodogram")

    y = r - np.nanmean(r)
    y = apply_window(y, window)
    n = y.size
    d = (x[1] - x[0]) if x.size > 1 else 1.0
    nfft = int(next_pow2(n) * max(1, int(pad_mul)))
    Y = np.fft.rfft(y, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=d)
    P = (np.abs(Y) ** 2) / n
    return freqs, P
