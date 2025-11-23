Resonant Medium Project
Full codebase & analysis pipelines for the Echo Equation cosmology framework

This repository contains the complete analysis environment used to develop and test the finite-memory cosmology model known as the Resonant Medium and its governing Echo Equation.
It includes:

Model solvers

BAO + SNe + CMB-lensing analysis pipelines

Figure-generation scripts (Figures 1–3 in the manuscript)

Memory-kernel and spectral-analysis tools

Phase-based development structure

Reproducibility utilities and guard-metric calculators

This repository accompanies:

Watts & Mal (2025), The Echo Equation: Finite-Memory Dynamics in Cosmology
(Submitted to JCAP)

Evidence Bundle (Primary Source)

All quantitative checks, figure-specific provenance files, guard metrics, and locked artifacts used in the manuscript are provided in the:

Echo Equation Evidence Bundle v1
Zenodo DOI: https://doi.org/10.5281/zenodo.17622475

A lightweight, referee-oriented mirror of this bundle is available here:

GitHub: https://github.com/Aikijase/echo-equation-evidence

These bundles contain:

SHA-256 lockfiles

machine-readable provenance for Figures 1–3

guard-metric summaries

compressed-likelihood inputs

phase-specific checks

tools/                # Analysis scripts, figure generators, utilities
paper-assembly/       # Paper-related scripts (not a submodule)
data/                 # Light supporting datasets (Pantheon+, BAO compressed files)
configs/              # Model + pipeline configuration files
phaseXX/              # Optional development phases
Tier3_1_EvidencePack_Phase21/   # Phase-21 evidence pack
tier3_evidence/                 # Additional evidence layers

Large observational datasets (e.g., full Planck maps, full Pantheon+ covariance)
are not stored in this repository due to size limits.
They are available via official archives and via the Zenodo evidence bundle.

Reproducing Figures from the Manuscript

Figure 1 (BAO + SNe comparison):

python tools/make_fig1_data.py
python tools/make_fig1_real.py


Figure 2 (Spectral density of resonant operator):

python tools/make_fig2_spectral.py


Figure 3 (CMB lensing predictions):

python tools/make_fig3_lensing.py


Additional reproduction notes are provided inside each tool script.

Citation

If you use this repository, the Echo Equation model, or the evidence bundle in your work, please cite:

Watts, J. & Mal, R. (2025),
"The Echo Equation: Finite-Memory Dynamics in Cosmology",
Echo Equation Evidence Bundle v1, Zenodo.
https://doi.org/10.5281/zenodo.17622475

License

MIT License.
See LICENSE in this repository.

Contact

Corresponding author:
Jason Watts
Email: theexperimentalistlab@outlook.com
