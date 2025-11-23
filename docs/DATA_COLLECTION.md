# Data Collection Guide — Cosmic Recycling (generated 2025-10-22T07:43:15)

This guide tells you **exactly what to pull** from each source and **where it goes** in the pack.

## Core Targets (do these first)
1. **ρ_BH(z)** — from **Shankar (2009)** review & related reconstructions  
   - Put values into `templates/rho_bh_z.csv` (z, rho_bh_Msun_Mpc3, sigma, method, source_id)
   - If only local ρ_BH is available, anchor z=0 and derive evolution via accretion check.

2. **Bolometric AGN LF** — from **Hopkins, Richards & Hernquist (2007)**  
   - Put z-binned (log10_Lbol, phi_dex) into `templates/agn_bolometric_lf.csv` with source_id=HOPKINS07
   - If the paper provides parametric fits only, sample a grid of logL per z from the fit.

3. **X-ray AGN LF** — from **Ueda et al. (2014)**  
   - Record (log10_Lx, phi_dex) and metadata; then convert to bolometric with `tools/data/band_to_bolometric.py`
   - Append or save as a second CSV and then combine with Hopkins LF as needed.

4. **Accretion History (ρ̇_acc)** — from the bolometric LF via Soltan argument  
   - Use `tools/data/agn_lf_to_accretion.py --epsilon 0.1`
   - Interpolate to the ρ_BH z-grid with `tools/data/interpolate_to_grid.py`
   - QC: `tools/data/qc_consistency_rhoBH_vs_accretion.py`

## Cross-checks
- **SFR history** — **Madau & Dickinson 2014** → `templates/bh_birthrate_models.csv`
- **GSMF** — **Shankar et al. 2016** (or other GSMF sources) → `templates/stellar_mass_function.csv`
- **H(z) & cosmology** — **DESI 2024**, **Planck 2018** → `templates/expansion_Hz.csv` and `templates/omega_m_z.csv`

## Provenance
For each row you add, ensure `source_id` is set and present in `docs/sources_catalog.csv` (DOIs prefilled).

## Command Cheatsheet
See `docs/nano_prompts.txt` for copy-paste commands covering validation, ingestion, conversions, interpolation, and QC.
