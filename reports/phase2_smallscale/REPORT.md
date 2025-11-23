# Phase‑2 Small‑Scale Suppression — Resonant Medium (Integrated)
**Goal:** Demonstrate that a resonant vacuum with memory (τ) and node spacing (ω)
can suppress dwarf‑scale structure (∼10⁹–10¹⁰ M⊙/h) while preserving ≥10¹¹ M⊙/h halos.

## What this module does
- Builds a toy ΛCDM linear P(k) (BBKS) and applies a **phase‑smoothed resonant modifier**:
  \[ P_{\rm res}(k) = P_{\Lambda {\rm CDM}}(k)\,\overline{\exp(-\tau k^2)\cos^2(\omega k + \phi)}_\phi \]
- Computes **σ(M)** and a **Halo Mass Function**, selectable:
  **Press–Schechter**, **Sheth–Tormen**, or **Tinker+08**.
- Produces **figures** (P(k), σ(M), HMF) and **JSON summaries**.
- Estimates **dwarf counts** by integrating the HMF over [10⁹,10¹⁰] M⊙/h
  within a Milky‑Way‑like virial sphere.

## How to run
```bash
# One-shot runner
bash tools/phase2_smallscale/run_phase2_smallscale.sh

# Or directly with config
python3 tools/phase2_smallscale/run.py --config tools/phase2_smallscale/config_smallscale.yaml

# Change HMF model in YAML: ps | st | tinker
```

## Outputs
- `outputs/phase2_smallscale/pk_compare.png`
- `outputs/phase2_smallscale/sigma_compare.png`
- `outputs/phase2_smallscale/hmf_compare.png`
- `outputs/phase2_smallscale/summary.json`
- `outputs/phase2_smallscale/combined_results.json` (includes dwarf counts)
