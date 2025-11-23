# SESSION NOTES (compact timeline)

- **BAO‑only fits**
  - Interleaved DM/DH vector works; with simple priors, LCDM baseline fits BAO well.
  - Resonant model sometimes improved χ² but at the cost of parameters; AIC/BIC generally **favored LCDM** in BAO‑only.

- **SN (Pantheon+) integration**
  - Using the **full STAT+SYS covariance** with a **MU_SH0ES** vector produced **χ² ≈ 1e10** → mismatch.
  - Switched to **standardized magnitudes `m_b_corr`** and **floored** the covariance to be SPD → stable Cholesky & finite χ².

- **Joint BAO+SN (full cov)**
  - LCDM baseline (alpha basis fixed; r_d fixed) runs cleanly.
  - Resonant with r_d fixed and modest priors is ready; compare via `tools/compare_ic.py`.
  - If exploring calibration freedom, enable `H0` in the fit with bounds.

- **Key takeaways for handover**
  - SN vector **must match** the covariance space.
  - Always verify cov is **SPD** before inversion (we included a tool to do this).
  - Use **information criteria** (AIC/BIC/AICc) to judge model parsimony.
