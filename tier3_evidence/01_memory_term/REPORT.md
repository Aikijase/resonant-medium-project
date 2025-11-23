# Phase-2 Mini-Report

_Generated: 2025-10-15T18:20:56_

## Growth (fσ8)
- χ²: **340.273**
- AIC: **340.273**
- BIC: **340.273**
- WWI: **0.0**
- Overlay plot: `plots/fs8_overlay.png`
- Residuals plot: `plots/fs8_residuals.png`

## ISW (placeholder)
- Bands: **270 ℓ-points**
- Note: placeholder spectra; scoring disabled.

## Joint Summary
- File: `outputs/phase2/pred_eval.phase2_joint.json`
- phase1_WWI: **None**
- fs8_WWI: **0.0**
- isw_info: ISW placeholder only (no data yet)

## Resonant Basins & Stability
- Heatmap: `plots/stability_heatmap.png`
- Basins (WWI contours + stability hatch): `plots/resonant_basins.png`
**Reading the figure:** color = WWI (% vs ΛCDM on fσ₈ with 1-param amplitude fit);
white contours at 50/70/85; hatched regions pass stability (bounded D, low TV_D, low stress).

## ISW amplitude (WISE + RACS)

- Combined χ²: **1.203** (n=4, k=0)
- AIC: **1.203**, ΔAIC vs LCDM(A=1): **0.645**
- WWI: **0.0**

| survey | z_eff | A_model | A_obs ± σ | resid |
|---|---:|---:|---:|---:|
| RACS_SKADS | 1.56 | 1.151 | 0.82 ± 0.36 | 0.92 |
| RACS_BACCUS | 1.56 | 1.151 | 0.94 ± 0.42 | 0.50 |
| WISE_GAL | 0.25 | 1.197 | 1.24 ± 0.47 | -0.09 |
| WISE_AGN | 1.10 | 1.106 | 0.88 ± 0.74 | 0.31 |

## fσ8 growth (Phase-2)
- χ²: **3.261** (n=5, k=1)
- AIC/BIC: **5.261 / 4.870**, ΔAIC/ΔBIC: **-19.667 / -20.057**
- WWI: **100.0**
- Amplitude fit α (S8-like): **14.686**
- Plot: `plots/fs8_overlay.png`, residuals: `plots/fs8_residuals.png`

## ISW amplitude (WISE + RACS)
- Combined χ²: **1.203** (n=4, k=0)
- AIC: **1.203**, ΔAIC vs LCDM(A=1): **0.645**
- WWI: **0.0**

| survey | z_eff | A_model | A_obs ± σ | resid |
|---|---:|---:|---:|---:|
| RACS_SKADS | 1.56 | 1.151 | 0.820 ± 0.360 | 0.921 |
| RACS_BACCUS | 1.56 | 1.151 | 0.940 ± 0.420 | 0.503 |
| WISE_GAL | 0.25 | 1.197 | 1.240 ± 0.470 | -0.092 |
| WISE_AGN | 1.10 | 1.106 | 0.880 ± 0.740 | 0.305 |

- Figure: `plots/isw_amplitude.png`

## Stability & Resonant Basins
- Stability heatmap: `plots/stability_heatmap.png`
- WWI contours + stability hatch: `plots/resonant_basins.png`
**Reading:** color = WWI (% vs ΛCDM on fσ8 with 1-param amplitude fit); white contours at 50/70/85; hatched regions pass stability (bounded D, low TV_D, low stress).


## Basins with ISW Compatibility
- Figure: `plots/resonant_basins_isw.png`
**Reading:** color = fs8 WWI; white = WWI {50,70,85} contours; hatched = stable; black outline = ISW-OK (χ²/n ≤ 2 using WISE+RACS amplitudes).

## fσ8 growth (Phase-2)
- χ²: **3.261** (n=5, k=1)
- AIC/BIC: **5.261 / 4.870**, ΔAIC/ΔBIC: **-19.667 / -20.057**
- WWI: **100.0**
- Amplitude fit α (S8-like): **14.686**
- Plot: `plots/fs8_overlay.png`, residuals: `plots/fs8_residuals.png`

## ISW amplitude (WISE + RACS)
- Combined χ²: **1.203** (n=4, k=0)
- AIC: **1.203**, ΔAIC vs LCDM(A=1): **0.645**
- WWI: **0.0**

| survey | z_eff | A_model | A_obs ± σ | resid |
|---|---:|---:|---:|---:|
| RACS_SKADS | 1.56 | 1.151 | 0.820 ± 0.360 | 0.921 |
| RACS_BACCUS | 1.56 | 1.151 | 0.940 ± 0.420 | 0.503 |
| WISE_GAL | 0.25 | 1.197 | 1.240 ± 0.470 | -0.092 |
| WISE_AGN | 1.10 | 1.106 | 0.880 ± 0.740 | 0.305 |

- Figure: `plots/isw_amplitude.png`

## Stability & Resonant Basins
- Stability heatmap: `plots/stability_heatmap.png`
- WWI contours + stability hatch: `plots/resonant_basins.png`
**Reading:** color = WWI (% vs ΛCDM on fσ8 with 1-param amplitude fit); white contours at 50/70/85; hatched regions pass stability (bounded D, low TV_D, low stress).

## Phase-3: Growth-rate evolution γ(z) = γ₀ + g₁ z/(1+z)
- Best: γ₀ = **1.400**, g₁ = **0.400**
- α (S8-like) = **1.555**, χ² = **37.916**
- AIC/BIC = **39.916 / 40.219**, WWI = **0.0**
- Heatmap: `plots/phase3_gammaevo_heatmap.png`
- Annotated: `plots/phase3_gammaevo_heatmap_annotated.png`

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |

## Phase-4: CMB lensing (κκ) & galaxy–lensing (g×κ)

### κκ (CMB lensing auto)
- n=1, χ²=0.069, ⟨|res|⟩≈0.26σ
- AIC=0.069, ΔAIC=0.069, WWI=0.0

### g×κ (galaxy–lensing, shape-only)
- n=5, χ²=0.621, ⟨|res|⟩≈0.35σ
- AIC=3.954, ΔAIC=0.000, WWI=0.0
- Â=0.993 ± 0.039 | A_pred=1.000 | pull=+0.17σ (Δχ²_pred=0.028)

| Catalog | z_eff | A_lit ± σ | Residual (σ) |
|---|---:|---:|---:|
| WISE×Planck-kappa | 0.35 | 0.990 ± 0.080 | -0.13 |
| NVSS×Planck-kappa | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES×Planck-kappa | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS×Planck-kappa | 0.65 | 0.940 ± 0.090 | -0.67 |
| RACS×Planck-kappa | 0.80 | 1.050 ± 0.150 | +0.33 |


## Phase-4: g×κ (multi-bin)

- Overall: n=9, ⟨|res|⟩≈0.32σ, ΔAIC=0.000, WWI=0.0
- Â=0.999 ± 0.032 | A_pred=1.000 | pull=+0.05σ

### Per-survey summary

| Survey | n | Â ± σ | ⟨|res|⟩ (σ) | ΔAIC | WWI |
|---|---:|---:|---:|---:|---:|
| DES | 1 | 1.000 ± 0.070 | 0.00 | 0.000 | 0.0 |
| KiDS | 1 | 0.940 ± 0.090 | 0.67 | 0.000 | 0.0 |
| NVSS | 1 | 1.020 ± 0.090 | 0.22 | 0.000 | 0.0 |
| RACS | 3 | 1.052 ± 0.094 | 0.33 | 0.000 | 0.0 |
| WISE | 3 | 0.994 ± 0.048 | 0.16 | 0.000 | 0.0 |

### Per-bin details

| Survey | Bin | z_eff | A_lit ± σ | Resid (σ) |
|---|---:|---:|---:|---:|
| WISE | 1 | 0.25 | 0.980 ± 0.090 | -0.22 |
| WISE | 2 | 0.35 | 0.990 ± 0.080 | -0.13 |
| WISE | 3 | 0.45 | 1.010 ± 0.080 | +0.13 |
| RACS | 1 | 0.60 | 1.040 ± 0.160 | +0.25 |
| RACS | 2 | 0.80 | 1.050 ± 0.150 | +0.33 |
| RACS | 3 | 1.00 | 1.070 ± 0.180 | +0.39 |
| NVSS | 1 | 1.00 | 1.020 ± 0.090 | +0.22 |
| DES | 1 | 0.55 | 1.000 ± 0.070 | +0.00 |
| KiDS | 1 | 0.65 | 0.940 ± 0.090 | -0.67 |

