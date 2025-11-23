#!/usr/bin/env python3
import csv, json, subprocess as sp, itertools, sys, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

ANALYZER = "tools/phase8_spectral/analyze_spectrum.py"

def _run_one(task):
    (mu0, nu0, tau, kappa, sigma8, Om, n_steps, zero_pad, stem) = task
    cmd = [
        "python3", ANALYZER,
        "--mu0", str(mu0), "--nu0", str(nu0),
        "--tau", str(tau), "--kappa", str(kappa),
        "--sigma8", str(sigma8), "--Om", str(Om),
        "--a-min", "1e-3", "--a-max", "1.0",
        "--n-steps", str(n_steps),
        "--zero-pad", str(zero_pad),
        "--quadratic-peak", "--lorentzian-fit",
        "--out-stem", str(stem),
    ]
    try:
        sp.run(cmd, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
        J = json.load(open(f"{stem}_summary.json"))
        lp = J.get("lorentzian", {})
        row = [
            tau, kappa,
            J.get("omega_at_max"), J.get("peak_power"),
            J.get("quad_peak_omega"), J.get("quad_peak_power"),
            lp.get("A"), lp.get("omega0"), lp.get("gamma"), lp.get("C"), lp.get("Q"),
            "ok",""
        ]
    except Exception as e:
        row = [tau, kappa, None, None, None, None, None, None, None, None, None, "err", str(e)]
    return (tau, kappa, row)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count()//2))
    # Tweak ranges here if you like
    ap.add_argument("--taus", type=str, default="0.10,0.20,0.30,0.40,0.60")
    ap.add_argument("--kappas", type=str, default="-0.05,-0.02,0.00,0.02,0.05")
    ap.add_argument("--n-steps", type=int, default=16000)
    ap.add_argument("--zero-pad", type=int, default=8)
    ap.add_argument("--mu0", type=float, default=0.99)
    ap.add_argument("--nu0", type=float, default=0.055)
    ap.add_argument("--sigma8", type=float, default=0.81)
    ap.add_argument("--Om", type=float, default=0.3)
    args = ap.parse_args()

    taus   = [float(s) for s in args.taus.split(",") if s.strip()]
    kappas = [float(s) for s in args.kappas.split(",") if s.strip()]

    out_csv = Path("outputs/phase8/sweep_tau_kappa.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for t in taus:
        for k in kappas:
            stem = Path(f"outputs/phase8/sweep_tau{t:.2f}_kap{k:+.2f}")
            tasks.append((args.mu0, args.nu0, t, k, args.sigma8, args.Om,
                          args.n_steps, args.zero_pad, str(stem)))

    print(f"[sweep] {len(tasks)} runs  |  --jobs={args.jobs}")
    t0 = time.time()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau","kappa","omega_at_max","peak_power",
                    "quad_peak_omega","quad_peak_power",
                    "lorentz_A","lorentz_omega0","lorentz_gamma","lorentz_C","lorentz_Q",
                    "status","note"])

    done = 0
    N = len(tasks)
    def update(res):
        nonlocal done
        done += 1
        tau, kappa, row = res
        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
        elapsed = time.time() - t0
        rate = done/elapsed if elapsed>0 else 0
        eta  = (N-done)/rate if rate>0 else 0
        sys.stdout.write(f"\r[{done:02d}/{N}] τ={tau:.2f} κ={kappa:+.2f}  "
                         f"elapsed {elapsed:5.1f}s  eta {eta:5.1f}s")
        sys.stdout.flush()
        if done == N: sys.stdout.write("\n")

    with Pool(processes=args.jobs) as pool:
        for res in pool.imap_unordered(_run_one, tasks):
            update(res)

    print(f"[wrote] {out_csv}")

if __name__ == "__main__":
    main()
