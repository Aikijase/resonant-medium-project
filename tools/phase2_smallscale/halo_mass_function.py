# halo_mass_function.py
import numpy as np

# Cosmology helpers (toy: BBKS transfer)
def bbks_transfer(k, Omega_m=0.3, h=0.7):
    Gamma = Omega_m * h
    q = k / Gamma
    T = np.log(1 + 2.34*q) / (2.34*q)
    T *= (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4) ** (-0.25)
    return T

def linear_pk(k, n_s=0.965, Omega_m=0.3, h=0.7):
    T = bbks_transfer(k, Omega_m=Omega_m, h=h)
    return (k**n_s) * (T**2)

# Windows
def W_tophat(x):
    W = np.ones_like(x)
    small = x < 1e-4
    xs = x[small]
    if xs.size > 0:
        W[small] = 1 - xs**2/10.0
    big = ~small
    xb = x[big]
    W[big] = 3*(np.sin(xb) - xb*np.cos(xb))/ (xb**3 + 1e-300)
    return W

def R_of_M(M, rho_m):
    return (3*M/(4*np.pi*rho_m))**(1/3)

def sigma_of_M(Pk, k, M, rho_m):
    R = R_of_M(M, rho_m)
    x = k * R
    W = W_tophat(x)
    integrand = (k**2) * Pk * (W**2)
    sig2 = np.trapz(integrand, k) / (2*np.pi**2)
    return float(np.sqrt(max(sig2, 1e-30)))

def sigma_grid(Pk, k, M_grid, rho_m):
    return np.array([sigma_of_M(Pk, k, M, rho_m) for M in M_grid])

# HMF variants -------------------
DELTA_C = 1.686

def hmf_press_schechter(M, sigma, rho_m):
    lnM = np.log(M)
    ln_sig_inv = np.log(1.0/ sigma)
    dln_sig_inv_dlnM = np.gradient(ln_sig_inv, lnM)
    nu = DELTA_C / sigma
    f = np.sqrt(2/np.pi) * nu * np.exp(-0.5*nu**2)
    dn = (rho_m / M) * f * np.abs(dln_sig_inv_dlnM)
    return np.where(np.isfinite(dn) & (dn>0), dn, np.nan)

def hmf_sheth_tormen(M, sigma, rho_m, A=0.3222, a=0.707, p=0.3):
    lnM = np.log(M)
    ln_sig_inv = np.log(1.0/ sigma)
    dln_sig_inv_dlnM = np.gradient(ln_sig_inv, lnM)
    nu = DELTA_C / sigma
    f = A * np.sqrt(2*a/np.pi) * nu * np.exp(-a*nu**2/2.0) * (1 + (nu**(-2*a*p)))
    dn = (rho_m / M) * f * np.abs(dln_sig_inv_dlnM)
    return np.where(np.isfinite(dn) & (dn>0), dn, np.nan)

def hmf_tinker08(M, sigma, rho_m, delta=200.0):
    """
    Simplified Tinker+08 mass function. We use their fitting form:
    f(σ) = A [ (σ/b)^(-a) + 1 ] exp(-c/σ^2)
    with coefficients vs overdensity delta=200 approximated.
    (For illustration; for precision, use a library like Colossus.)
    """
    # Coeffs for Δ=200 (approx; static z=0 values)
    A, a, b, c = 0.186, 1.47, 2.57, 1.19
    lnM = np.log(M)
    ln_sig_inv = np.log(1.0/ sigma)
    dln_sig_inv_dlnM = np.gradient(ln_sig_inv, lnM)
    f = A * ( (sigma/b)**(-a) + 1.0 ) * np.exp(-c / (sigma**2))
    dn = (rho_m / M) * f * np.abs(dln_sig_inv_dlnM)
    return np.where(np.isfinite(dn) & (dn>0), dn, np.nan)
