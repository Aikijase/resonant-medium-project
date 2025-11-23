# Tier‑3: Dusty Plasma Dispersion — Reconstructed Sets

These are reconstructed dispersion curves using published parameter values, *not* digitised plot points.

## Files
- dusty_ring_dispersion_reconstructed.csv
- dusty_ring_dispersion.png
- dusty_2d_crystal_dispersion_reconstructed.csv
- dusty_2d_crystal_dispersion.png

## 1-D Ring (Sheridan & Gallagher 2010)
Params: omega0=16.6 rad/s, alpha=1.41, kappa_bar=1.32. 
Model: nearest-neighbor Yukawa chain with a=1.
Longitudinal: omega_L^2 ~ (4 * omega0^2 * C) sin^2(ka/2), C = exp(-kappa_bar*alpha)*(1+kappa_bar*alpha)/alpha^3.
Transverse: omega_T^2 ~ (gap*omega0)^2 + (4 * omega0^2 * Ct) sin^2(ka/2), gap=0.35, Ct=0.3*C.

## 2-D Crystal (Nunomura et al. 2002)
Isotropic long-wavelength approx:
omega_L^2 = omega_E^2 + Cs^2 k^2 + beta k^4
omega_T^2 = omega_T0^2 + Ct^2 k^2
with omega_E=22.0, omega_T0=8.0, Cs=10.0, Ct=6.0, beta=8.0.

## Caveats
- k is dimensionless (a=1). Rescale if lattice spacing a known.
- Replace with author-provided raw tables when available; keep both versions for transparency.
