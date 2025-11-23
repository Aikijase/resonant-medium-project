# tools/phase4_summary.py
#!/usr/bin/env python3
"""
Standalone Phase-4 summary printer (κκ + g×κ).
Safe to call from tools/run_phase2.sh without modifying your existing summary.
"""
import json, math, os

def safe_load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

def print_kappa_kappa():
    J = safe_load("outputs/phase4/lensing_score.json")
    if not J: 
        return
    n = J.get("n", 1)
    chi2 = float(J.get("chi2", 0.0))
    mean_sigma = (chi2 / max(n,1))**0.5
    print("CMB lensing κκ")
    print(f"  χ²={chi2:.3f}  n={n}  ⟨|res|⟩≈{mean_sigma:.2f}σ")
    try:
        print(f"  AIC={J['AIC']['value']:.3f}  ΔAIC={J['AIC']['delta']:.3f}  WWI={J.get('WWI')}")
    except Exception:
        pass
    print()

def print_gxk():
    J = safe_load("outputs/phase4/gk_score.json")
    if not J:
        return
    n = int(J.get("n", 0))
    chi2 = float(J.get("chi2", 0.0))
    mean_sigma = (chi2 / max(n,1))**0.5 if n else float("nan")
    print("Galaxy–lensing g×κ (shape-only)")
    print(f"  χ²={chi2:.3f}  n={n}  ⟨|res|⟩≈{mean_sigma:.2f}σ")
    try:
        print(f"  AIC={J['AIC']['value']:.3f}  ΔAIC={J['AIC']['delta']:.3f}  WWI={J.get('WWI')}")
    except Exception:
        pass
    # Extra diagnostics
    try:
        Ahat = J["A_hat"]; sA = J["sigma_A_hat"]; Ap = J["scale_model"]
        dchi = J["delta_chi2_pred"]
        pull = (Ap - Ahat)/sA
        print(f"  Â={Ahat:.3f} ± {sA:.3f}   A_pred={Ap:.3f}   pull={pull:+.2f}σ   (Δχ²_pred={dchi:.3f})")
    except Exception:
        pass
    # Per-catalog lines
    D = J.get("details", [])
    for d in D:
        try:
            print(f"    • {d['name']}: z_eff={d['z_eff']:.2f}  A_lit={d['A_lit']:.3f}±{d['sigma_lit']:.3f}  resid={d['resid_sigma_model']:+.2f}σ")
        except Exception:
            continue
    print()

def main():
    print("=== Phase-4 Summary ===")
    print_kappa_kappa()
    print_gxk()

if __name__ == "__main__":
    main()
