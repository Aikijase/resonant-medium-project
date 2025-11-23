# Troubleshooting

- **Gigantic χ² (~1e10)** → SN vector (MU) used with full STAT+SYS cov. Fix: build the SN vector from `m_b_corr` (standardized mags).
- **Cholesky fails** → raise `--floor-frac` in `make_floored_cov.py` (e.g., `3e-5` or `1e-4`).
- **Bounds parsing error** → pass as one token: `--A-bounds "-2,2"` or `--A-bounds=-2,2`.
- **Missing files** → confirm `PROJ` path and `data/...` layout in the one‑shot snippet.
