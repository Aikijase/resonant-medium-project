#!/usr/bin/env python3
"""
Phase-4 multi-bin summary printer (reads outputs/phase4/gk_multi.json).
Shows overall + per-survey lines.
"""
import json, math, os

PATH = "outputs/phase4/gk_multi.json"

def fmt(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except: return "—"

def main():
    if not os.path.exists(PATH):
        print("[warn] no outputs/phase4/gk_multi.json"); return
    J = json.load(open(PATH))
    n = J["n"]
    mean_sigma = math.sqrt(J["chi2_pred"]/max(n,1))
    pull = (J["scale_model"] - J["A_hat"]) / J["sigma_A_hat"]
    print("=== Phase-4 Multi-bin g×κ Summary ===")
    print(f"Overall: n={n}  ⟨|res|⟩≈{mean_sigma:.2f}σ  ΔAIC={fmt(J['AIC']['delta'])}  WWI={J['WWI']}")
    print(f"  Â={fmt(J['A_hat'])} ± {fmt(J['sigma_A_hat'])}   A_pred={fmt(J['scale_model'])}   pull={pull:+.2f}σ")
    print()
    print("Per-survey:")
    for sname, S in sorted(J["surveys"].items()):
        ms = math.sqrt(S["chi2_pred"]/max(S["n"],1))
        pull_s = (J["scale_model"] - S["A_hat"]) / S["sigma_A_hat"]
        print(f"  • {sname}: n={S['n']}  ⟨|res|⟩≈{ms:.2f}σ  ΔAIC={fmt(S['AIC']['delta'])}  WWI={S['WWI']}  Â={fmt(S['A_hat'])}±{fmt(S['sigma_A_hat'])}  pull={pull_s:+.2f}σ")

if __name__ == "__main__":
    main()

