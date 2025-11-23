
# Resonant Medium Project — Phase 2 Bundle

**Goal:** Extend the ε–γ universality collapse into quantum/fluid analogues (BEC Hawking, SBSL, superfluid/optical cavities) and wire it to the cosmic anchor picture.

## Contents

- `registry_BEC.yaml` — template entries for adding BEC Hawking-ridge datasets to your existing `registry.yaml` flow.
- `digitize_bec_ridge.py` — quick digitizer: click points on published ridge figures to export CSVs.
- `collapse_compare_all.py` — reads normalized CSVs and overlays a universal collapse; fits ε(ω̂) = A[1 - exp(-(ω̂/w0)^p)].
- `fit_cosmic_anchor.py` — scaffolding to connect ε(ω̂) to a cosmic “anchor frequency” via ρ_BH(z) or horizon-scale mapping.
- `run_phase2.sh` — suggested commands for a clean run.
- `git_quickstart.sh` — fix identity + initial commit helper.

> Assumptions: CSV schema is simple: `omega, omega0, epsilon, gamma(optional), label(optional)`.
> If your existing pipeline writes different column names, adjust the small mapping dictionaries in the scripts.

## Quick Start

```bash
# 1) Digitize a BEC ridge plot → produce bec_ridge_raw.csv
python3 digitize_bec_ridge.py --image path/to/steinhauer_fig.png --out data/bec_ridge_raw.csv

# 2) Normalize to omega_hat = omega/omega0 and keep epsilon
#    (digitizer already writes columns: omega, omega0, epsilon, label)
#    If you have multiple ridges: run digitize multiple times, then concat.

# 3) Overlay + Fit universal law (A, w0, p)
python3 collapse_compare_all.py   --inputs data/bec_ridge_raw.csv data/abh_plate_norm.csv data/dusty_plasma_norm.csv   --out-prefix outputs/universal_collapse

# 4) (Optional) Cosmic layer
python3 fit_cosmic_anchor.py   --bh-density data/rho_BH_z.csv   --map omega(z)=c/R_H(z)   --out outputs/cosmic_anchor_fit.json
```

## Notes

- The digitizer uses `matplotlib.ginput` (click to add points, Enter to finish). Save multiple traces as needed.
- The fitter uses `scipy.optimize.curve_fit`. Provide `--bounds` if your data prefers tighter priors.
- The plotting functions avoid specifying colors/styles, per your environment constraints.
- All outputs land in `outputs/`. Adjust as you like.
