#!/usr/bin/env bash
set -euo pipefail
cd ~/resonant-medium-project
source .venv/bin/activate
mkdir -p outputs

# Joint: chi2 vs phi
python3 - <<'PY'
import glob, json, re, numpy as np, math
import matplotlib.pyplot as plt
LCDM = 794.3161378478268
rows=[]
for p in glob.glob("outputs/joint_phase_phi*.json"):
    m=re.search(r'phi([0-9.]+)\.json$', p)
    if not m: continue
    phi=float(m.group(1))
    J=json.load(open(p))
    chi=J["chi2"]; br=J["breakdown"]; A=J["best_params"].get("A",float('nan'))
    rows.append((phi, chi, chi-LCDM, br["chi2_bao"], br["chi2_sn"], A))
rows=sorted(rows)
phis=np.array([r[0] for r in rows]); chis=np.array([r[1] for r in rows])
imin=chis.argmin(); chi_min=chis[imin]
mask = chis <= chi_min + 2.0
a,b,c = np.polyfit(phis[mask], chis[mask], 2)
phi_best = -b/(2*a)
sigma_phi = math.sqrt(1/a) if a>0 else float('nan')
phi_line = np.linspace(min(phis)-0.05, max(phis)+0.05, 300)
chi_fit  = a*phi_line**2 + b*phi_line + c
plt.figure(figsize=(6,4.2))
plt.scatter(phis, chis, label="runs")
plt.plot(phi_line, chi_fit, label="quadratic near min")
plt.axhline(chi_min+1.0, linestyle="--", linewidth=1, label="Δχ²=+1")
plt.title("Joint: $\\chi^2$ vs phase $\\phi$ at fixed $f=3.26$")
plt.xlabel("$\\phi$ [rad]"); plt.ylabel("$\\chi^2$")
plt.legend(title=f"min ≈ {chi_min:.3f} at φ ≈ {phi_best:.3f} ± {sigma_phi:.3f}")
plt.tight_layout(); plt.savefig("outputs/fig_joint_chi2_vs_phi.png", dpi=150)
print("Wrote outputs/fig_joint_chi2_vs_phi.png")
PY

# BAO-only: chi2_bao vs f
python3 - <<'PY'
import glob, json, re, numpy as np, math
import matplotlib.pyplot as plt
BAO_BASE = 41.09367841682559
rows=[]
for p in glob.glob("outputs/bao_only_profile_f_*.json"):
    m=re.search(r'_f_([0-9.]+)\.json$', p)
    if not m: continue
    f=float(m.group(1))
    J=json.load(open(p))
    chi_bao = J["breakdown"]["chi2_bao"]
    rows.append((f, chi_bao))
rows=sorted(rows)
fs=np.array([r[0] for r in rows]); chis=np.array([r[1] for r in rows])
imin=chis.argmin(); chi_min=chis[imin]
mask = chis <= chi_min + 2.0
a,b,c = np.polyfit(fs[mask], chis[mask], 2)
f_best = -b/(2*a)
sigma_f = math.sqrt(1/a) if a>0 else float('nan')
f_line = np.linspace(min(fs)-0.02, max(fs)+0.02, 400)
chi_fit = a*f_line**2 + b*f_line + c
plt.figure(figsize=(6,4.2))
plt.scatter(fs, chis, label="runs")
plt.plot(f_line, chi_fit, label="quadratic near min")
plt.axhline(chi_min+1.0, linestyle="--", linewidth=1, label="Δχ²=+1")
plt.title("BAO-only: $\\chi^2_{\\rm BAO}$ vs frequency $f$")
plt.xlabel("$f$"); plt.ylabel("$\\chi^2_{\\rm BAO}$")
plt.legend(title=f"min ≈ {chi_min:.3f} at f ≈ {f_best:.3f} ± {sigma_f:.3f}")
plt.tight_layout(); plt.savefig("outputs/fig_baoonly_chi2_vs_f.png", dpi=150)
print("Wrote outputs/fig_baoonly_chi2_vs_f.png")
PY

# Note file (only creates if missing)
if [ ! -f outputs/RESULT_NOTE.md ]; then
  cat > outputs/RESULT_NOTE.md <<'MD'
(placeholder) See earlier command to write the full note.
MD
  echo "Wrote outputs/RESULT_NOTE.md (placeholder)"
fi
