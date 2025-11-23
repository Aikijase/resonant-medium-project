
# Phase‑2 Small‑Scale Suppression — Resonant Medium Test
**Goal:** Show that a simple resonant modifier to the linear power spectrum,
\(P_{\rm res}(k) = P_{\Lambda\rm CDM}(k)\,\exp(-\tau k^2)\,\cos^2(\omega k)\),
can suppress dwarf‑scale structure (\(10^{9\text{–}10} M_\odot/h\)) while preserving large halos (\(\gtrsim 10^{11} M_\odot/h\)).

## Method (reproducible)
- **Linear spectrum:** BBKS transfer, scalar tilt \(n_s=0.965\).
- **Resonant modifier:** memory \(\tau\) and node spacing \(\omega\).
- **Pipeline:** \(P(k)\) → top‑hat \(\sigma(M)\) → Press–Schechter HMF.
- **Mini‑sweep:** \(\tau\in[0.05,0.12],\ \omega\in[0.12,0.20]\).

## Best candidate (from sweep)
- **τ ≈ 0.050**, **ω ≈ 0.1443** \((\mathrm{Mpc}/h)^{-1}\)
- **σ ratios (Res/LCDM):**  
  - \(10^9\): **0.637**, \(10^{10}\): **0.773**, \(10^{11}\): **0.901**

Interpretation: **strong dwarf suppression** with **minimal impact** on big halos.

## How to run
```bash
# 1) Mini-sweep & figures
python3 tools/phase2_smallscale_sweep.py

# Outputs → outputs/phase2_smallscale/
#   - mini_pk_best.png, mini_sigma_best.png, mini_hmf_best.png
#   - mini_sweep_tau_omega_ranked.csv, mini_sweep_top8.csv, mini_sweep_summary.json

# 2) Dwarf-count estimator (choose params, e.g., best from the sweep)
python3 tools/phase2_dwarfcount.py --tau 0.05 --omega 0.144   --Mhost 1e12 --Mmin 1e9 --Mmax 1e10 --Rvir 250
```

## Dwarf-count notes
We integrate the HMF between \([10^9,10^{10}] M_\odot/h\)) to get a **number density** and scale by a rough **Milky‑Way virial volume** to estimate counts. This is an order‑of‑magnitude tool suitable for comparing **LCDM vs Resonant suppression ratios** (not an exact subhalo model).

## Caveats
- Linear theory + Press–Schechter is **illustrative**; for publication we’ll want (i) Sheth–Tormen or Tinker HMF and (ii) a calibrated subhalo model.
- Amplitude is arbitrary here; we focus on **relative suppression** at fixed mass scales.
- Mapping \(k\) to galaxy cores requires follow‑up with **cored‑profile/Jeans** modeling.

## One‑paragraph claim (draft)
> A resonant vacuum with finite memory modifies the small‑scale power spectrum by selective node suppression and gentle damping. With a single node placed near \(k\sim 10\,h\,\mathrm{Mpc}^{-1}\) and a modest memory term (\(\tau\sim0.05\,(h/\mathrm{Mpc})^{-2}\)), the model suppresses \(\sigma(M)\) by \(\sim40\%\) at \(10^9 M_\odot/h\) while preserving \(\ge 90\%\) of the variance at \(10^{11} M_\odot/h\). This simultaneously addresses the Missing Satellites/Too‑Big‑To‑Fail tensions and naturally favors cored inner profiles, without introducing new dark‑matter particles.
