import json, pathlib
p = pathlib.Path("outputs/phase2")
fs8  = json.load(open(p/"fs8_eval.json"))
isw  = json.load(open(p/"isw_eval.json"))
join = json.load(open(p/"pred_eval.phase2_joint.json"))

def line(k,v): print(f"{k:<18} {v}")

print("\n=== Phase-2 Summary ===")
line("fs8 χ²",        f"{fs8['chi2']:.3f}")
line("fs8 AIC",       f"{fs8['AIC']['value']:.3f}")
line("fs8 BIC",       f"{fs8['BIC']['value']:.3f}")
line("fs8 WWI",       f"{fs8.get('WWI')}")
line("ISW points",    sum(len(b["ell"]) for b in isw["bandpowers"]))
line("Joint file",    str(p/"pred_eval.phase2_joint.json"))
print("Artifacts:")
for f in ["fs8_eval.json","fs8_eval.csv","isw_eval.json","pred_eval.phase2_joint.json"]:
    print(" -", p/f)
for f in ["plots/fs8_overlay.png","plots/fs8_residuals.png"]:
    print(" -", f)
print()
# --- Phase-4: CMB lensing (kappa-kappa) ---
try:
    J = json.load(open("outputs/phase4/lensing_score.json"))
    n = J.get("n", 1)
    chi2 = J["chi2"]
    mean_sigma = (chi2 / max(n,1))**0.5
    print(f"CMB lensing κκ χ²  {chi2:.3f}  (n={n}, ⟨|res|⟩≈{mean_sigma:.2f}σ)")
    print(f"CMB lensing AIC    {J['AIC']['value']:.3f}")
    print(f"CMB lensing ΔAIC   {J['AIC']['delta']:.3f}")
    print(f"CMB lensing WWI    {J.get('WWI')}")
except Exception:
    pass

# --- Phase-4: galaxy–lensing (g×κ) ---
try:
    J = json.load(open("outputs/phase4/gk_score.json"))
    n = J.get("n", 0)
    chi2 = J["chi2"]
    mean_sigma = (chi2 / max(n,1))**0.5 if n else float("nan")
    print(f"g×κ shape-only χ²  {chi2:.3f}  (n={n}, ⟨|res|⟩≈{mean_sigma:.2f}σ)")
    print(f"g×κ shape-only AIC {J['AIC']['value']:.3f}")
    print(f"g×κ shape-only ΔAIC {J['AIC']['delta']:.3f}")
    print(f"g×κ shape-only WWI {J.get('WWI')}")
except Exception:
    pass
