# Resonant-Medium Cosmology: Data Plan

This file summarizes the *minimum viable dataset* to test the 'dark matter as medium, dark energy as resonance' toy model against state-of-the-art observations. Start with a ΛCDM baseline fit, then turn on the new parameters and compare Δχ² (AIC/BIC).

## Phases
1. **Baseline reproduction** — Fit ΛCDM to Pantheon+ + DESI BAO + (optional) Planck compressed priors. Verify we match published posteriors.
2. **Add growth** — Include fσ8 (RSD) + one shear set (KiDS-1000 or HSC-Y3). Check growth vs expansion consistency.
3. **Model extension** — Introduce resonant field parameters: (α, β, k, …). Fit expansion-only; then full (expansion+growth+shear).
4. **Predictions** — Compute H(z), D_M(z), fσ8(z) residual patterns; look for oscillatory features implied by resonance.

## Folder Layout
data/
  planck2018/…
  pantheon_plus/…
  desi_dr1_bao/…
  des_y3/…
  kids_1000/…
  hsc_y3/…
  growth_fsigma8/…
code/
  loaders/*.py
  models/resonant_medium.py
  likelihoods/*.py
  run_fits.py
outputs/
  chains/
  corner_plots/
  metrics/

## Likelihood Blocks (minimal)
- SN: Pantheon+ binned (m_B vector, cov).
- BAO: DESI DR1 distance measurements with cov.
- Growth: fσ8(z) points with cov (or diagonal σ if cov not available).
- Shear: start with one survey’s 2pt (data vector + cov + nuisance priors).
- Priors: Ω_b h² (BBN), H0 (optional SH0ES), flat priors on resonance params.

## KPIs
- ΔAIC / ΔBIC vs ΛCDM.
- Posterior on resonance frequency / coupling (non-zero?).
- Goodness-of-fit per probe (SN-only, BAO-only, Growth-only, Joint).
- Tension metrics (S8, H0) with/without resonance parameters.

