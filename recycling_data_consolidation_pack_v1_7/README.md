# Cosmic Recycling — Data Consolidation Pack (v1)

**Created:** 2025-10-22T07:20:42

This pack gives you academically traceable scaffolding to collect, validate, and consolidate datasets for the Cosmic Recycling ODE + Predictions pipeline.

## Structure
- `templates/` — CSV templates you will fill with real data (z-ordered, explicit units).
- `schema/` — YAML-like schema notes for each template (columns, units, constraints).
- `tools/data/` — ingest + validate stubs (drop-in scripts; do **not** patch existing code).
- `tools/` — phase runner + prediction sheet maker stubs.
- `logs/` — master log for checkpoints & validation runs.
- `docs/` — publication traceability ledger.

## Quick Start
1. Fill the CSVs in `templates/` with your sources (see `docs/sources_catalog.csv` for citations).
2. Run:
   ```
   python3 tools/data/validate_pack.py templates/ schema/ logs/master_log.csv
   ```
3. When green, consolidate:
   ```
   python3 tools/data/ingest_rho_bh.py templates/rho_bh_z.csv --out data/rho_bh_z.clean.csv
   ```
   (Repeat for other templates.)
4. Build predictions sheet:
   ```
   python3 tools/make_predictions_sheet.py --pack-root .
   ```

All files are self-documented with inline headers for reproducibility.
