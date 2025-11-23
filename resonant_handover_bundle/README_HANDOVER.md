# Resonant Medium — Full Handover (2025-10-01)

This bundle gives you **everything a fresh chat/session needs** to pick up your BAO+SN work:
- Copy‑paste **one-shot commands**
- Small, robust utilities (Python + Bash)
- Clear **path conventions**, pitfalls, and **troubleshooting**
- A quick **glossary** and **session notes**

It assumes your repo lives at:
```
$PROJ = $HOME/resonant-medium-project
```
If not, just change `PROJ` in the commands.

---

## ⚡ TL;DR: Why things broke & how we fixed them

- Using **Pantheon+ full STAT+SYS covariance** requires the **standardized magnitude** vector `m_b_corr` (not the SH0ES‑anchored `MU_SH0ES`). Mismatch ⇒ enormous χ² (~1e10).
- The published STAT+SYS covariance can be **ill‑conditioned**. We **eigen‑floor** it to be SPD: `Pantheon+SH0ES_STAT+SYS.fixed_spd_floored.csv`.
- BAO inputs should be in **long/interleaved** format with columns `kind,z,y_data,sigma` (kind ∈ {DM,DH}).

Utilities in `tools/` automate all of this.

---

## ✅ One‑shot: reproduce LCDM & Resonant (full SN covariance)

Open a fresh terminal, then paste **exactly** (edit paths only if different):

```bash
set -euo pipefail

PROJ="$HOME/resonant-medium-project"
BAO_CSV="$PROJ/data/desi_dr1_bao/bao_measurements_long_interleaved_sigma.csv"
BAO_COV="$PROJ/data/desi_dr1_bao/bao_covariance_plus_lya.csv"

# Pantheon+ catalog & cov (CSV paths)
SN_CAT="$PROJ/data/pantheon_plus/Pantheon+SH0ES.csv"
SN_DAT="$PROJ/data/pantheon_plus/Pantheon+SH0ES.dat"
SN_COV_TXT="$PROJ/data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.cov"
SN_COV_FLOOR="$PROJ/data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.fixed_spd_floored.csv"

SN_DIR="$PROJ/data/sn"
SN_VEC="$SN_DIR/sn_pantheonplus_mb_MATCHED.csv"

OUT_LCDM="$PROJ/outputs/joint_lcdm_Pplus_FULLCOV_mb.json"
OUT_RESN="$PROJ/outputs/joint_resonant_rd_fixed_Pplus_FULLCOV_mb.json"

LOG_LCDM="$PROJ/outputs/lcdm_fullcov_mb.log"
LOG_RESN="$PROJ/outputs/resn_fullcov_mb.log"

mkdir -p "$SN_DIR" "$PROJ/outputs"

# [A] If the CSV catalog is missing, convert the .dat once
if [ ! -f "$SN_CAT" ] && [ -f "$SN_DAT" ]; then
  python3 - <<'PY'
import pandas as pd, pathlib as p, re
PROJ = p.Path.home()/ "resonant-medium-project"
dat = PROJ/"data/pantheon_plus/Pantheon+SH0ES.dat"
csv = PROJ/"data/pantheon_plus/Pantheon+SH0ES.csv"
df  = pd.read_csv(dat, sep=r"\s+", comment="#", engine="python")
df.to_csv(csv, index=False)
print("[ok] wrote", csv)
PY
fi

# [B] Ensure floored SPD covariance (idempotent)
python3 "/mnt/data/handover_bundle/tools/make_floored_cov.py"   --in  "$SN_COV_TXT"   --out "$SN_COV_FLOOR"   --floor-frac 1e-5

# [C] Build SN vector that matches the full cov (uses m_b_corr)
python3 "/mnt/data/handover_bundle/tools/sn_normalize.py"   --in  "$SN_CAT"   --out "$SN_VEC"   --use-mb-corr   --zcol zHD

# [D] LCDM baseline (alpha basis fixed; r_d fixed)
stdbuf -oL -eL python3 -u "$HOME/sn_bao_joint_alpha_clean.py"   --bao-csv "$BAO_CSV"   --bao-cov "$BAO_COV"   --sn-csv  "$SN_VEC"   --sn-cov  "$SN_COV_FLOOR"   --use-alpha-basis --fid-H0 70.0 --fid-Om 0.3 --fid-rd 147.1   --A-bounds 0,0 --f-bounds 1,1 --phi-bounds 0,0 --gamma-bounds 0,0   --rd-bounds 147.1,147.1   --fit A f phi gamma rd   --out "$OUT_LCDM" | tee "$LOG_LCDM"

# [E] Resonant model (r_d fixed; modest priors; multi‑restart)
stdbuf -oL -eL python3 -u "$HOME/sn_bao_joint_alpha_clean.py"   --bao-csv "$BAO_CSV"   --bao-cov "$BAO_COV"   --sn-csv  "$SN_VEC"   --sn-cov  "$SN_COV_FLOOR"   --use-alpha-basis --fid-H0 70.0 --fid-Om 0.3 --fid-rd 147.1   --rd-bounds 147.1,147.1   --prior-A-sigma 0.4   --f-bounds 0.3,6.0 --gamma-bounds 0,1.5   --restarts 8   --fit A f phi gamma rd   --out "$OUT_RESN" | tee "$LOG_RESN"

# [F] Compare IC metrics (lower is better)
python3 "/mnt/data/handover_bundle/tools/compare_ic.py"   --lcdm "$OUT_LCDM"   --resn "$OUT_RESN"
```

**Dials you can tweak later**:
- Allow absolute calibration freedom:
  ```bash
  --H0-bounds 60,85 --fit H0 A f phi gamma rd
  ```
- If Cholesky still fails, raise `--floor-frac` to `3e-5` or `1e-4` in step **[B]**.

---

## 🧰 What’s in `tools/`

- `sn_normalize.py` — creates a **matched SN vector** for the full STAT+SYS cov (uses `m_b_corr`, `zHD`). Can also emit MU for diag‑only work.
- `make_floored_cov.py` — makes the Pantheon+ covariance **SPD** by eigen‑flooring, prints condition numbers and Cholesky status.
- `compare_ic.py` — prints **AIC/BIC/AICc/χ²** for two JSON results.

All are CLI tools with `--help`.

---

## 🧪 BAO Input Expectations

`sn_bao_joint_alpha_clean.py` expects BAO in **long/interleaved** form:
```
kind,z,y_data,sigma
DM,0.51,  D_M/r_d value,  sigma
DH,0.51,  D_H/r_d value,  sigma
DM,0.706, ...
DH,0.706, ...
...
```
If you pass a BAO covariance, `sigma` is only used for sanity checks; the cov dominates.

---

## 🧱 Troubleshooting (you’ve seen these)

- **χ² ≈ 1e10** → SN vector doesn’t match full covariance. Use `--use-mb-corr` to build `sn_pantheonplus_mb_MATCHED.csv`.
- **Covariance not SPD** → increase `--floor-frac` in `make_floored_cov.py` until Cholesky succeeds.
- **`--A-bounds: expected one argument`** → pass bounds as a **single token**: `--A-bounds "-2,2"` or `--A-bounds=-2,2`.
- **`FileNotFoundError`** → verify paths in the one‑shot block and create missing folders.
- **Pandas “bottleneck” warning** → harmless (performance only).

---

## 📓 Session Notes (short)

See `docs/SESSION_NOTES.md` for a compact timeline & outcomes that a new chat can scan in 1–2 minutes.
See also `docs/GLOSSARY.md` for definitions.
