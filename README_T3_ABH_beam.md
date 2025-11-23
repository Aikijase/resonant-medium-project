# Tier‑3 Dataset: Acoustic Black Hole (ABH) Beam — Power‑Law Taper

This package provides synthetic, literature‑aligned data for a mechanical Acoustic Black Hole (ABH) beam,
computed with an Euler–Bernoulli finite‑difference model (variable stiffness EI(x) and mass per length μ(x)).

## Model
- Beam length L = 0.500 m, width b = 0.020 m
- Material: steel (E = 2.000e+11 Pa, ρ = 7850 kg/m³)
- Reference thickness at x=L: h0 = 0.0040 m
- Taper: h(x) = h0 · (x/L)^m with m ∈ {0 (uniform), 2, 3, 4}
- Boundary: Cantilever (clamped at x=0, free at x=L)
- PDE: (EI w'')'' = μ ω² w, EI(x)=E b h(x)^3/12, μ(x)=ρ b h(x)

## Files
- abh_profiles.csv                 # x vs thickness for m = 0,2,3,4
- abh_frequencies.csv              # first four natural frequencies for each m
- abh_thickness_profiles.png       # visual check of h(x)
- abh_mode_frequencies.png         # frequency trends vs mode for each m
- abh_mode_shape_m0_vs_m3.png      # normalized first‑mode shape comparison

## Usage
- Include this folder under T3 evidence: datasets/ABH_beam/
- Cite classical ABH literature (Mironov 1988; Krylov 1994; Pelat 2012).

## Notes
- This is a demo‑grade solver intended to show **trend‑level ABH effects** (soft-edge energy sink).
- Real ABH beams include a small truncation layer and sometimes viscoelastic coating; we clamp h(x) with a tiny epsilon.
