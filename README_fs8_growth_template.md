# fσ8 Growth Measurements — Data Template

This CSV is a clean template for compiling fσ8(z) growth measurements.

## Columns
- `z`: Effective redshift of the measurement
- `fs8`: Measured value of f(z) * σ8(z)
- `sigma_fs8`: 1σ uncertainty
- `survey`: Survey/instrument (e.g., 6dFGS, 2dF, BOSS DR12, eBOSS DR14Q, VIPERS, WiggleZ, etc.)
- `method`: RSD (redshift-space distortions), PV (peculiar velocities), etc.
- `AP_corrected`: 'yes' if the Alcock–Paczynski correction has been applied (as in standard curated lists)
- `kmax_or_scale`: Optional scale-cut marker (k_max or r_min used in analysis)
- `cov_note`: Note if this point is part of a correlated set or if a covariance matrix is provided
- `reference`: ArXiv ID or DOI of the source
- `year`: Publication year
- `notes`: Anything useful for later cross-checks

## Suggested Source Sets
- Sagredo, Nesseris & Sapone (2018) — *Internal Robustness of Growth Rate data*, Phys. Rev. D 98, 083543 — vetted 22-point set.
- “Gold-2017” subset referenced in Nesseris et al. and used in many analyses.
- Manna & Desai (2024) — EPJC — 23-point set (Gold-2017 + eBOSS DR14Q + BOSS DR12 CMASS).
- Kazantzidis & Perivolaropoulos (2018) — 63-point extended compilation (older+newer RSD data; caution about correlations).

## How to Use
1. Open the sources, copy their tables, and enter rows into this CSV.
2. Keep `AP_corrected='yes'` for curated lists that already apply the AP fix.
3. If a paper provides a covariance for multiple points, set `cov_note='has_cov'` for those rows and keep the covariance file alongside this CSV.
4. Save as `data/fs8_growth.csv` inside your project when done.

