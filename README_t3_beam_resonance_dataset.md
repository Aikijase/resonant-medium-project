# Tier‑3 Dataset: Euler–Bernoulli Beam Resonance (Demo)

This dataset provides fundamental and higher‑mode resonance frequencies for slender beams,
computed with the Euler–Bernoulli model for common materials, sizes, lengths, and boundary conditions.

## Formula
For a uniform, slender beam with rectangular section:
- Area: A = b·h
- Second moment: I = b·h³ / 12
- Natural frequency (Hz) for mode *n*:
  
  fₙ = (λₙ² / (2π L²)) · √(E I / (ρ A))

where:
- E = Young’s modulus (Pa)
- ρ = density (kg/m³)
- L = length (m)
- λₙ = dimensionless constant set by boundary condition

## λₙ values used (first four modes)
- Cantilever (fixed‑free): [1.87510407, 4.69409113, 7.85475744, 10.99554073]
- Simply supported: [π, 2π, 3π, 4π]
- Fixed–fixed: [4.73004074, 7.85320462, 10.99560784, 14.13716839]

## Columns
- material, E_Pa, rho_kg_m3
- b_m, h_m, A_m2, I_m4
- L_m
- boundary_condition
- mode, lambda_n
- frequency_hz

## Notes
- Valid for slender beams where shear deformation/rotary inertia are negligible (Euler–Bernoulli regime).
- Units are SI.
