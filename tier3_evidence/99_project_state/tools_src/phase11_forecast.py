#!/usr/bin/env python3
"""
Phase-11: Partial-lock forecasts from Phase-10 weights.

Reads:   outputs/phase10/surface.json   (w_tau0, w_kappa0, w_A0)
Makes:   outputs/phase11/forecast.csv   (pct, tau, kappa, A)
         outputs/phase11/forecast.png   (Δ vs % reduction)
         outputs/phase11/status.txt     (PASS/FAIL summary)

Model: linear in % reduction: Δ ≈ w_lock * (reduction_pct / 100)
PASS if: w_A0 > w_tau0 > w_kappa0  (strict ordering)
"""
import os, json, csv
import matplotlib.pyplot as plt

SURFACE = "outputs/phase10/surface.json"
OUTDIR  = "outputs/phase11"
PCTS    = [0,10,20,30,40,50,60,70,80]

def main():
    J = json.load(open(SURFACE))
    w = J.get("weights", {})
    w_tau  = float(w.get("w_tau0", 0.0))
    w_kap  = float(w.get("w_kappa0", 0.0))
    w_A    = float(w.get("w_A0", 0.0))

    os.makedirs(OUTDIR, exist_ok=True)

    # CSV
    rows = []
    for p in PCTS:
        rows.append({
            "pct": p,
            "tau":   w_tau * (p/100.0),
            "kappa": w_kap * (p/100.0),
            "A":     w_A   * (p/100.0),
        })
    with open(f"{OUTDIR}/forecast.csv","w",newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["pct","tau","kappa","A"])
        wtr.writeheader(); wtr.writerows(rows)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot([r["pct"] for r in rows], [r["tau"]   for r in rows], label="τ reduction")
    plt.plot([r["pct"] for r in rows], [r["kappa"] for r in rows], label="κ reduction")
    plt.plot([r["pct"] for r in rows], [r["A"]     for r in rows], label="A reduction")
    plt.xlabel("% reduction")
    plt.ylabel("Δ (predicted)")
    plt.title("Phase-11: Δ vs % reduction (from Phase-10 weights)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/forecast.png", dpi=200)

    # PASS / FAIL
    ordering_ok = (w_A > w_tau > w_kap)
    with open(f"{OUTDIR}/status.txt","w") as f:
        f.write("Phase-11 status\n")
        f.write(f"w_tau0   = {w_tau:.6g}\n")
        f.write(f"w_kappa0 = {w_kap:.6g}\n")
        f.write(f"w_A0     = {w_A:.6g}\n")
        f.write(f"ORDER    = A > τ > κ  -> {'PASS' if ordering_ok else 'FAIL'}\n")
    print("Wrote", f"{OUTDIR}/forecast.csv")
    print("Wrote", f"{OUTDIR}/forecast.png")
    print("Wrote", f"{OUTDIR}/status.txt")
    print("RESULT:", "PASS ✅ (A>τ>κ)" if ordering_ok else "FAIL ❌ (ordering not A>τ>κ)")

if __name__ == "__main__":
    main()
