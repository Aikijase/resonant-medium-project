#!/usr/bin/env python3
"""
Quick, defensible conclusion for Phase-8 resonance:
- Baseline (tau=0, kappa=0)
- Memory: tau in {0.20, 0.30, 0.40}, kappa=-0.02
Criteria:
  PASS if:
   1) Baseline has near-DC blip only: omega* < 0.1 and no Lorentzian
   2) ≥2 memory cases have a coherent peak:
        - Lorentzian present
        - |omega0 - quad_peak|/omega0 <= 0.2
        - Prominence >= 0.15 (A / (A + C))
   3) Memory omega0 values are within 20% of each other (stable band)
Outputs:
  - Prints a table + verdict
  - Writes outputs/phase8/quick_conclusion.md
"""
import json, subprocess as sp, os
from pathlib import Path
from statistics import median

ANALYZER = "tools/phase8_spectral/analyze_spectrum.py"
OUTDIR = Path("outputs/phase8")
OUTDIR.mkdir(parents=True, exist_ok=True)

def run(mu0,nu0,tau,kappa,s8,Om,stem):
    cmd = [
        "python3", ANALYZER,
        "--mu0", str(mu0), "--nu0", str(nu0),
        "--tau", str(tau), "--kappa", str(kappa),
        "--sigma8", str(s8), "--Om", str(Om),
        "--a-min", "1e-3", "--a-max", "1.0",
        "--n-steps", "16000",
        "--zero-pad", "8",
        "--quadratic-peak", "--lorentzian-fit",
        "--out-stem", str(stem)
    ]
    sp.run(cmd, check=True, stdout=sp.PIPE, stderr=sp.PIPE)

def read_summary(stem):
    J = json.load(open(f"{stem}_summary.json"))
    omega = J.get("omega_at_max")
    quad  = J.get("quad_peak_omega", omega)
    L = J.get("lorentzian")
    if L:
        A = L.get("A", 0.0); C = L.get("C", 0.0)
        omega0 = L.get("omega0", None)
        gamma  = L.get("gamma", None)
        Q      = L.get("Q", None)
        prom   = (A/(A+C)) if (A is not None and C is not None and (A+C)>0) else None
    else:
        omega0 = gamma = Q = prom = None
    return dict(omega=omega, quad=quad, omega0=omega0, gamma=gamma, Q=Q, prom=prom, L=bool(L))

def pct(a,b): 
    return abs(a-b)/a if (a and b) else float("inf")

def main():
    cases = []
    # Baseline
    base_stem = OUTDIR/"qc_no_memory"
    run(0.99,0.055,0.0,0.0,0.81,0.3,str(base_stem))
    cases.append(("baseline", 0.0, 0.0, read_summary(str(base_stem))))

    # Memory set
    mem_params = [(0.20,-0.02),(0.30,-0.02),(0.40,-0.02)]
    for tau,kappa in mem_params:
        stem = OUTDIR/f"qc_mem_tau{tau:.2f}_kap{kappa:+.2f}"
        run(0.99,0.055,tau,kappa,0.81,0.3,str(stem))
        cases.append((f"tau={tau:.2f},kappa={kappa:+.2f}", tau, kappa, read_summary(str(stem))))

    # Evaluate criteria
    base = cases[0][3]
    base_ok = (base["omega"] is not None and base["omega"] < 0.1 and not base["L"])

    mem = cases[1:]
    mem_good = []
    for name, tau, kap, R in mem:
        coherent = (R["L"] and R["omega0"] is not None)
        close_to_quad = (coherent and R["quad"] and pct(R["omega0"], R["quad"]) <= 0.2)
        prominent = (coherent and R["prom"] is not None and R["prom"] >= 0.15)
        mem_good.append((name, coherent, close_to_quad, prominent, R))
    n_pass = sum(1 for _,c1,c2,c3,_ in mem_good if (c1 and c2 and c3))

    # Stability across memory cases
    ovals = [R["omega0"] for _,c1,c2,c3,R in mem_good if (c1 and R["omega0"])]
    stable = False
    band_pct = None
    if len(ovals) >= 2:
        med = median(ovals)
        band_pct = max(abs(o-med) for o in ovals)/med if med>0 else float("inf")
        stable = band_pct <= 0.20  # 20% band

    PASS = base_ok and n_pass >= 2 and stable

    # Print table
    print("\n=== Quick Conclusion Report ===")
    print(f"Baseline ok? {base_ok} (omega*={base['omega']:.6f}, lorentz={base['L']})")
    print("\nMemory cases:")
    print("name\t\tomega*\tquad\tomega0\tgamma\tQ\tprom\tcoherent/near-quad/prom")
    for name, _, _, R in cases[1:]:
        coh = "Y" if R["L"] else "N"
        nq  = "Y" if (R["L"] and R["quad"] and pct(R["omega0"],R["quad"])<=0.2) else "N"
        pr  = "Y" if (R["prom"] is not None and R["prom"]>=0.15) else "N"
        print(f"{name:16s}\t{(R['omega'] or float('nan')):.6f}\t{(R['quad'] or float('nan')):.6f}\t"
              f"{(R['omega0'] or float('nan')):.6f}\t{(R['gamma'] or float('nan')):.6f}\t"
              f"{(R['Q'] or float('nan')):.3f}\t{(R['prom'] or float('nan')):.3f}\t{coh}/{nq}/{pr}")
    print(f"\nStability across memory cases: {stable} (band ~{(band_pct*100 if band_pct is not None else float('nan')):.2f}%)")
    print("\nVERDICT:", "PASS ✅ Resonant feature detected & stable" if PASS else "INCONCLUSIVE ⚠️ needs more data")

    # Write markdown summary
    md = OUTDIR/"quick_conclusion.md"
    with open(md, "w") as f:
        f.write("# Phase-8 Quick Conclusion\n\n")
        f.write(f"- Baseline ok: **{base_ok}** (omega*={base['omega']:.6f}, lorentz={base['L']})\n")
        f.write(f"- Memory stability band: **{stable}** (±{(band_pct*100 if band_pct is not None else float('nan')):.2f}%)\n")
        f.write(f"- Memory cases passing coherence/near-quad/prominence: **{n_pass}/3**\n")
        f.write(f"- **Verdict:** {'PASS ✅ Resonant feature detected & stable' if PASS else 'INCONCLUSIVE ⚠️ needs more data'}\n\n")
        f.write("## Details\n\n")
        f.write("| case | omega* | quad | omega0 | gamma | Q | prom |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name,_,_,R in cases[1:]:
            f.write(f"| {name} | {R['omega'] or float('nan'):.6f} | {R['quad'] or float('nan'):.6f} | "
                    f"{R['omega0'] or float('nan'):.6f} | {R['gamma'] or float('nan'):.6f} | "
                    f"{R['Q'] or float('nan'):.3f} | {R['prom'] or float('nan'):.3f} |\n")
    print(f"\n[wrote] {md}")
    print("Done.")

if __name__ == "__main__":
    main()
