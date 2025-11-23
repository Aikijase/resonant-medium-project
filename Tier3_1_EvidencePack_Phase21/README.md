Phase-21 Evidence Pack — Tier 3.1

This directory contains the Tier-3.1 evidence generated during Phase-21 of the Resonant Medium Project.
Phase-21 tests the closure, dark coupling, and memory-viscosity properties of the finite-memory cosmology model using the high-precision reconstruction pipeline.

The metrics below are extracted from the locked Phase-21 run included in the Echo Equation Evidence Bundle v1 (Zenodo DOI: https://doi.org/10.5281/zenodo.17622475
).

Key Results
Memory Kernel Parameters

ε* = 0.014313
(Effective memory-strength estimator: governs the amplitude of the resonant correction.)

η_dark = 5.45 × 10⁻⁶
(Dark-sector coupling term extracted from the viscosity-corrected operator.)

Closure Test

Closure = 1.000056

A perfect closure (1.000000) would indicate that the reconstructed operator exactly matches the forward model under finite-memory dynamics.
A value of 1.000056 corresponds to a 0.0056% deviation, well within the theoretical tolerance used for model validation (±0.01–0.02%).

This is considered a PASS.

Growth-Amplitude Constraint

σ₈₀ = 0.86

χ² = 33.77

Using the Phase-21 operator, the model predicts a late-time growth amplitude consistent with observational priors.
The χ² value reflects the fit quality from the growth-test pipeline applied to the viscosity-adjusted resonant operator.

Included Materials

The Phase-21 Tier-3.1 evidence pack includes:

Tier3.1_Phase21_Report.pdf     # Full report describing the Phase-21 test
Tier3_1_EvidencePack_Phase21   # Raw evidence structures + metrics
thresholds.json                # Threshold definitions for pass/fail
t3_beam_resonance_dataset.csv  # Dataset used in the Phase-21 resonance test


(Additional logs and intermediate files may appear as required by the reproducibility pipeline.)

Relation to Main Project

Phase-21 provides a high-precision, late-pipeline validation of the finite-memory operator.
It forms part of the model’s Tier-3 validation suite, which includes:

Tier-1: Sanity + structure tests

Tier-2: Kernel stability tests

Tier-3: Closure, resonance, and growth-amplitude tests

Phase-21 specifically validates that the resonant operator:

Maintains internal mathematical consistency (closure).

Produces observationally viable growth amplitudes.

Preserves the expected dark-coupling scale.

This evidence is referenced in the main paper’s Extended Validation section.

Citation

If you use Phase-21 results, please cite:

Watts, J. & Mal, R. (2025),
"The Echo Equation: Finite-Memory Dynamics in Cosmology",
Echo Equation Evidence Bundle v1, Zenodo.
https://doi.org/10.5281/zenodo.17622475
