# Figure Digitization & Extraction Guide (generated 2025-10-22T07:52:09)

This guide covers **Option B (Table + Figure Digitization)** for:
- Aird et al. (2015) — XLF (absorbed & unabsorbed)
- Ueda et al. (2014) — XLF + absorption (LDDE; CTK treatment)
- Hopkins et al. (2006) — Bolometric QLF

## Tools
- WebPlotDigitizer (WPD): https://apps.automeris.io/wpd/
- Use **log–log axes** mode for L and φ; input tick values carefully.
- Export CSV with columns `X, Y` in **data units** (not pixel units).

## Workflow (per figure/panel)
1. Open the figure in WPD. Set **X = log10(L)**, **Y = log10(φ)** when appropriate.
2. Calibrate axes using at least two ticks per axis. Double-check decades.
3. Digitize points **by z-slice** (colour/marker) — keep panels separate.
4. Export CSV. Then paste rows into the corresponding worksheet:
   - `templates/aird2015_xlf_digitize.csv`
   - `templates/ueda2014_xlf_digitize.csv`
   - `templates/hopkins2006_bolqlf_digitize.csv`
5. Fill metadata: `z_bin_center/low/high`, `band`, `absorbed_flag`, `fig_panel`, `source_id`.

## Units & Conventions
- X-ray LF: use **log10 L_X [erg s^-1]** and **φ [Mpc^-3 dex^-1]**.
- Bolometric QLF: **log10 L_bol [erg s^-1]**, **φ [Mpc^-3 dex^-1]**.
- If panel shows **φ per log10**, record φ directly; if per ln, convert: φ_dex = φ_ln / ln(10).
- Uncertainties: if only visual, use a conservative fractional error (e.g., 0.15 dex) and note it.

## Suggested z-slices to capture
- Low-z: 0.1, 0.3, 0.5
- Mid-z: 0.7, 1.0, 1.5, 2.0
- High-z: 2.5, 3.0, 3.5, 4–5 (as available)
Capture ≥10–12 L-bins across the break each z-slice if possible.

## After Digitization
Run validation and merge:
```
python3 tools/data/validate_pack.py templates/ schema/ logs/master_log.csv
python3 tools/data/merge_lf_sources.py   --aird templates/aird2015_xlf_digitize.csv   --ueda templates/ueda2014_xlf_digitize.csv   --hopkins templates/hopkins2006_bolqlf_digitize.csv   --out data/agn_lf_combined.csv
```
Then (optional) convert **X-ray LF → bolometric**:
```
python3 tools/data/band_to_bolometric.py data/agn_lf_combined.csv --out data/agn_lf_bolometric.csv
```
Finally compute **ρ̇** from bolometric LF and QC vs ρ_BH:
```
python3 tools/data/agn_lf_to_accretion.py data/agn_lf_bolometric.csv --epsilon 0.1 --out data/bh_accretion_from_lf.csv
python3 tools/data/qc_consistency_rhoBH_vs_accretion.py --rho-bh data/rho_bh_z.clean.csv --rho-dot data/bh_accretion_from_lf.csv --out logs/qc_rhoBH_vs_accretion.csv
```
