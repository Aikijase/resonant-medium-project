#!/usr/bin/env python3
import argparse, json, os, time, math
import numpy as np
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def load_bao_interleaved(bao_csv: str, bao_cov: str|None=None):
    import os, numpy as np, pandas as pd
    df = pd.read_csv(bao_csv)
    # Accept either ['z','y','sigma'] or ['z','y_data','sigma']
    z = df['z'].to_numpy(float)
    y = (df['y'] if 'y' in df.columns else df['y_data']).to_numpy(float)
    sigma = df['sigma'].to_numpy(float) if 'sigma' in df.columns else None
    C = None
    if bao_cov and os.path.exists(bao_cov):
        if bao_cov.endswith('.npy'):
            C = np.load(bao_cov)
        else:
            # try CSV (comma), then whitespace
            try:
                C = np.loadtxt(bao_cov, delimiter=',')
            except Exception:
                C = np.loadtxt(bao_cov)
    return z, y, sigma, C
# ---------- Numerics helpers ----------
def robust_cholesky(M, jitter0=0.0, max_tries=6):
    """Return lower-triangular L such that L L^T ≈ M, adding diagonal jitter if needed."""
    M = np.array(M, float)
    M = 0.5 * (M + M.T)
    jitter = jitter0
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(M + jitter * np.eye(M.shape[0]))
        except np.linalg.LinAlgError:
            jitter = max(1e-12, (10.0**k) * 1e-12)
    raise np.linalg.LinAlgError("Cholesky failed: matrix not SPD even with jitter.")

def parse_bounds(s):
    if s is None or s == "":
        return None
    a, b = s.split(",")
    return (float(a), float(b))

def hit_bound(val, bounds, tol=1e-9):
    if bounds is None:
        return False
    lo, hi = bounds
    return abs(val - lo) < tol or abs(val - hi) < tol

# ---------- Cosmology (flat, resonant w) ----------
c_over_H0 = 2997.92458  # Mpc for H0 normalized to 100 (c/H0, with H0 in 100 km/s/Mpc)

def w_res(z, A, f, phi, gamma):
    # Smooth, benign oscillatory DE around -1
    return -1.0 + A * np.cos(f * np.log1p(z) + phi) * np.exp(-gamma * z)

def dark_energy_factor(z, A, f, phi, gamma, nz=400):
    """
    ρ_DE(z)/ρ_DE(0) = exp[ 3 * ∫_0^z (1 + w(z'))/(1+z') dz' ].
    """
    if z <= 0.0:
        return 1.0
    zz = np.linspace(0.0, z, max(50, nz))
    integrand = (1.0 + w_res(zz, A, f, phi, gamma)) / (1.0 + zz)
    I = np.trapz(integrand, zz)
    return float(np.exp(3.0 * I))

def E_of_z_scalar(z, Om, A, f, phi, gamma, nz_int=300):
    Or = 0.0
    Ok = 0.0
    Ode0 = 1.0 - Om - Or - Ok
    de = dark_energy_factor(z, A, f, phi, gamma, nz=nz_int)
    return math.sqrt(Om * (1.0 + z) ** 3 + Ode0 * de)

def build_background_grid(zmax, H0, Om, A, f, phi, gamma, nz=800):
    z = np.linspace(0.0, float(zmax), max(200, nz))
    Ez = np.array([E_of_z_scalar(t, Om, A, f, phi, gamma, nz_int=300) for t in z])
    invE = 1.0 / Ez
    Dc = c_over_H0 / (H0 / 100.0) * np.trapz(invE, x=z) * 0.0  # placeholder to init
    # cumulative integral for Dc(z)
    Dc_vals = np.zeros_like(z)
    accum = 0.0
    for i in range(1, len(z)):
        accum += 0.5 * (invE[i] + invE[i - 1]) * (z[i] - z[i - 1])
        Dc_vals[i] = c_over_H0 / (H0 / 100.0) * accum
    Ez_at = lambda zz: np.interp(np.asarray(zz, float), z, Ez)
    Dc_at = lambda zz: np.interp(np.asarray(zz, float), z, Dc_vals)
    return z, Ez, Dc_at, Ez_at

# ---------- SN loader + χ² ----------
def load_sn_mu_and_cov(sn_csv: str, sn_cov: str | None, cov_kind: str, scale: float):
    df = pd.read_csv(sn_csv)
    if not {"z", "mu"}.issubset(df.columns):
        raise ValueError("SN CSV must contain columns: z, mu, (optional sigma).")

    z = df["z"].to_numpy(float)
    mu = df["mu"].to_numpy(float)

    if sn_cov and os.path.exists(sn_cov):
        if sn_cov.endswith(".npy"):
            M = np.load(sn_cov)
        else:
            M = pd.read_csv(sn_cov, header=None).values
        if M.shape != (len(z), len(z)):
            raise ValueError(f"SN matrix shape {M.shape} does not match N={len(z)}")

        if cov_kind == "prec":
            P = scale * M
            R = robust_cholesky(P)  # R R^T = P
            L_or_R = R
            is_prec = True
        elif cov_kind == "cov":
            C = M / (scale if scale > 0 else 1.0)
            L = robust_cholesky(C)  # L L^T = C
            L_or_R = L
            is_prec = False
        elif cov_kind == "auto":
            # Heuristic: if diagonal magnitudes look like precisions (>1), treat as precision
            d = np.diag(M).astype(float)
            if np.median(d) > 1.0:
                P = scale * M
                R = robust_cholesky(P)
                L_or_R = R
                is_prec = True
            else:
                C = M / (scale if scale > 0 else 1.0)
                L = robust_cholesky(C)
                L_or_R = L
                is_prec = False
        else:
            raise ValueError("sn-cov-kind must be one of auto,cov,prec")
    else:
        # Diagonal-only from 'sigma' column
        if "sigma" not in df.columns:
            raise ValueError("No SN covariance provided and 'sigma' missing from CSV.")
        sig = df["sigma"].to_numpy(float)
        C = np.diag(sig * sig)
        L_or_R = robust_cholesky(C)
        is_prec = False

    return z, mu, L_or_R, is_prec



def make_sn_chi2(z, mu, L, is_prec, H0, Om0=None, A0=0.0, f0=0.0, phi0=0.0, gamma0=0.0, nz_bg=800):
    """
    Returns chi2(theta) that reads parameters from theta each call.
    theta layout we expect during the joint fit: [A, f, phi, gamma, (Om?), (rd?)]
    SN ignores rd, but we allow it in theta so indexing stays aligned with BAO.
    """
    import numpy as np
    from numpy.linalg import solve

    z = np.asarray(z, float)
    mu = np.asarray(mu, float)

    def _unpack(theta):
        A,f,phi,gamma = A0, f0, phi0, gamma0
        Om = Om0 if Om0 is not None else 0.3
        i = 0
        if len(theta) >= 4:
            A,f,phi,gamma = theta[0:4]; i = 4
        if len(theta) >= i+1:
            Om = theta[i]; i += 1
        # optional rd ignored for SN
        return A,f,phi,gamma,Om

    def _quad(res):
        if is_prec:
            y = L @ res        # L^T L = P (precision)
        else:
            y = solve(L, res)  # L^T L = C (covariance)
        return float(y @ y)

    def chi2(theta):
        A,f,phi,gamma,Om = _unpack(theta)
        zmax = float(np.max(z)) + 0.1
        _, Dc, Dc_at, _ = build_background_grid(zmax, H0, Om, A, f, phi, gamma, nz=nz_bg)
        Dl = (1.0 + z) * Dc_at(z)
        mu_th = 5.0*np.log10(np.clip(Dl, 1e-12, None)) + 25.0
        return _quad(mu - mu_th)

    return chi2

def make_bao_chi2_both(bao_pack, H0, Om0, rd0, fit_Om=False, fit_rd=False, nz_bg=800):
    # --- DEBUG: inspect incoming bao_pack and mask ---
    try:
        y   = np.asarray(bao_pack.get("y", bao_pack.get("y_data")))
        C   = np.asarray(bao_pack.get("C", bao_pack.get("cov")))
        z   = np.asarray(bao_pack.get("z"))
        kind= np.asarray(bao_pack.get("kind")).astype(str) if "kind" in bao_pack else None
        print(f"[BAO DEBUG] len(y)={len(y) if y is not None else 'NA'}  "
              f"C.shape={None if C is None else C.shape}  "
              f"has z={z is not None}  has kind={kind is not None}  "
              f"unique kind={None if kind is None else sorted(set(kind.tolist()))}",
              flush=True)
    except Exception as _e:
        print(f"[BAO DEBUG] exception while introspecting bao_pack: {_e}", flush=True)

    """
    Accepts:
      - dict style: {"DM": (z, y, sigma, C), "DH": (z, y, sigma, C)}   (either key may be absent)
      - tuple/list: (z, y, sigma, C)  -> treated as a single DM/rd dataset
    Returns a chi2(theta) that reads resonant params if present.
    """
    import numpy as np
    from numpy.linalg import cholesky

    def _chi2(res, sigma=None, C=None):
        if C is not None:
            C = 0.5*(C + C.T)
            L = cholesky(C + 1e-12*np.eye(C.shape[0]))
            y = np.linalg.solve(L, res)
            return float(y @ y)
        elif sigma is not None:
            w = 1.0/np.asarray(sigma, float)
            return float(((res*w)**2).sum())
        else:
            return float((res**2).sum())

    def _unpack_theta(theta):
        # Default to no-resonant if not provided
        A=f=phi=gamma = 0.0, 0.0, 0.0, 0.0
        Om, rd = Om0, rd0
        i = 0
        n = len(theta)
        # If resonant present, they come first in the vector: [A,f,phi,gamma, (Om?), (rd?)]
        if n >= 4:
            A, f, phi, gamma = theta[0:4]; i = 4
        # Then optional Om, rd depending on fit flags
        if fit_Om and i < n:
            Om = theta[i]; i += 1
        if fit_rd and i < n:
            rd = theta[i]; i += 1
        return A, f, phi, gamma, Om, rd

    # Normalize into a dict-style object
    if isinstance(bao_pack, dict):
        packs = {}
        for k in ("DM","DH"):
            if k in bao_pack and bao_pack[k] is not None:
                z,y,sigma,C = bao_pack[k]
                packs[k] = (np.asarray(z,float), np.asarray(y,float), sigma, C)
    else:
        # Single dataset -> assume DM/rd
        z,y,sigma,C = bao_pack
        packs = {"DM": (np.asarray(z,float), np.asarray(y,float), sigma, C)}

    def chi2(theta):
        A, f, phi, gamma, Om, rd = _unpack_theta(theta)
        zmax = 0.0
        for k,(z,_,_,_) in packs.items():
            zmax = max(zmax, float(np.max(z)))
        zmax += 0.1
        _, Dc, Dc_at, H_at = build_background_grid(zmax, H0, Om, A, f, phi, gamma, nz=nz_bg)

        total = 0.0
        # DM contribution: y_model = D_M/rd = D_C/rd for flat geo
        if "DM" in packs:
            z, y_data, sigma, C = packs["DM"]
            y_model = Dc_at(z)/rd
            total += _chi2(y_data - y_model, sigma=sigma, C=C)

        # DH contribution: y_model = D_H/rd = c/H(z)/rd
        if "DH" in packs:
            z, y_data, sigma, C = packs["DH"]
            y_model = (2997.92458/(H0/100.0)) * (1.0/H_at(z)) / rd
            total += _chi2(y_data - y_model, sigma=sigma, C=C)

        return total

    return chi2
