#!/usr/bin/env python3
import json, subprocess as sp
from pathlib import Path
from statistics import median

ANALYZER = "tools/phase8_spectral/analyze_spectrum.py"
OUT = Path("outputs/phase8"); OUT.mkdir(parents=True, exist_ok=True)

def run(stem, tau, kappa, mu0=0.99, nu0=0.055, s8=0.81, Om=0.3):
    cmd = ["python3", ANALYZER,
           "--mu0", str(mu0), "--nu0", str(nu0),
           "--tau", str(tau), "--kappa", str(kappa),
           "--sigma8", str(s8), "--Om", str(Om),
           "--a-min", "1e-3", "--a-max", "1.0",
           "--n-steps", "16000", "--zero-pad", "8",
           "--quadratic-peak", "--lorentzian-fit",
           "--out-stem", str(stem)]
    sp.run(cmd, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
    J = json.load(open(stem.as_posix()+"_summary.json"))
    L = J.get("lorentzian") or {}
    prom = (L.get("A")/(L.get("A")+L.get("C"))) if L and (L.get("A") is not None) and (L.get("C") is not None) and (L.get("A")+L.get("C")>0) else None
    return dict(stem=stem.name, tau=tau, kappa=kappa,
                omega=J.get("omega_at_max"),
                quad=J.get("quad_peak_omega") or J.get("omega_at_max"),
                omega0=L.get("omega0"), gamma=L.get("gamma"), Q=L.get("Q"),
                prom=prom, hasL=bool(L))

def rel_spread(a, b):
    """Symmetric relative spread between two positive numbers."""
    return abs(a-b)/((a+b)/2.0)

def main():
    # Baseline
    base = run(OUT/"real_base_tau0_kap0", tau=0.0, kappa=0.0)

    # Memory (tau fixed), probe three kappas
    mem = [
        run(OUT/"real_tau0p30_kap-0p05", tau=0.30, kappa=-0.05),
        run(OUT/"real_tau0p30_kap-0p02", tau=0.30, kappa=-0.02),
        run(OUT/"real_tau0p30_kap+0p02", tau=0.30, kappa=+0.02),
    ]

    base_ok = (base["omega"] is not None and base["omega"] < 0.1 and not base["hasL"])

    # Keep only coherent memory cases with decent prominence
    mem_coh = [m for m in mem if m["hasL"] and m["omega0"] and (m["prom"] or 0) >= 0.15]

    # Regime-aware stability: if ANY PAIR of coherent cases are within 20%, call that a stable band
    stable_pair = None
    min_pair_spread = None
    for i in range(len(mem_coh)):
        for j in range(i+1, len(mem_coh)):
            wi, wj = mem_coh[i]["omega0"], mem_coh[j]["omega0"]
            r = rel_spread(wi, wj)
            if (min_pair_spread is None) or (r < min_pair_spread):
                min_pair_spread = r
                stable_pair = (mem_coh[i], mem_coh[j])
    stability_ok = (min_pair_spread is not None and min_pair_spread <= 0.20)

    # Also detect regime split (factor > ~2 between min/max ω0 among coherent)
    regime_split = False
    if len(mem_coh) >= 2:
        ws = [m["omega0"] for m in mem_coh]
        regime_split = (max(ws)/min(ws) >= 2.0)

    verdict = "PASS ✅" if (base_ok and stability_ok) else "INCONCLUSIVE ⚠️"

    print("\n=== Real-Physics Quick Check (regime-aware) ===")
    print(f"Baseline OK? {base_ok}  (omega*={base['omega']:.6f}, Lorentz={base['hasL']})")
    print("\nMemory cases (tau=0.30):")
    print("name\t\tkappa\tomega*\tquad\tomega0\tgamma\tQ\tprom\tcoherent")
    for m in mem:
        print(f"{m['stem']:20s}\t{m['kappa']:+.2f}\t{(m['omega'] or float('nan')):.6f}\t{(m['quad'] or float('nan')):.6f}\t"
              f"{(m['omega0'] or float('nan')):.6f}\t{(m['gamma'] or float('nan')):.6f}\t{(m['Q'] or float('nan')):.3f}\t"
              f"{(m['prom'] or float('nan')):.3f}\t{m['hasL']}")

    print(f"\nCoherent memory cases: {len(mem_coh)}/{len(mem)}")
    if stable_pair:
        a, b = stable_pair
        print(f"Best stable pair: κ={a['kappa']:+.2f} vs κ={b['kappa']:+.2f}  "
              f"spread≈{min_pair_spread*100:.1f}%  (ω0≈{a['omega0']:.3f} vs {b['omega0']:.3f})")
    print(f"Regime split detected: {regime_split}")
    print("\nVERDICT:", verdict)

if __name__ == "__main__":
    main()
