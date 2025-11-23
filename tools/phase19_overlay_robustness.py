#!/usr/bin/env python3
import argparse, csv, json, subprocess, sys
from pathlib import Path

def run_si(w2, K, steps=8000, burn=300, noise=0.01, kv=0.10, kx=0.20, eps=0.06, adapt=15, preset="neuron"):
    p = subprocess.run([sys.executable, "tools/phase10_phasecouple_demo.py",
        "--preset", preset, "--omega2", str(w2), "--kv", str(kv), "--kx", str(kx),
        "--Kphi", str(K), "--eps", str(eps), "--adapt_every", str(adapt),
        "--noise", str(noise), "--steps", str(steps), "--burn_in", str(burn),
        "--prefix", f"p19r_w{w2:.3f}_K{K:.3f}"], capture_output=True, text=True)
    return json.loads(p.stdout)["metrics"]["sync_index"]

def local_robust(w2c, Kc, dw=0.01, dK=0.005, span_w=0.04, span_k=0.02, steps=8000, si_thresh=0.95):
    # tiny grid around curve point
    import numpy as np
    w_vals = np.arange(w2c - span_w/2, w2c + span_w/2 + 1e-9, dw)
    k_vals = np.arange(Kc - span_k/2, Kc + span_k/2 + 1e-9, dK)
    ok=0; tot=0
    for w in w_vals:
        for K in k_vals:
            si = run_si(float(w), float(K), steps=steps)
            ok += (si >= si_thresh)
            tot += 1
    return ok / max(tot,1)

def main():
    ap = argparse.ArgumentParser(description="Phase-19.1 Overlay robustness on edge curve")
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--outdir", default="outputs/phase19")
    ap.add_argument("--prefix", default="p19_edgeR")
    ap.add_argument("--sample-step", type=int, default=5, help="use every Nth point to keep runtime small")
    ap.add_argument("--dw", type=float, default=0.01)
    ap.add_argument("--dK", type=float, default=0.005)
    ap.add_argument("--span-w", type=float, default=0.04)
    ap.add_argument("--span-k", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--si-thresh", type=float, default=0.95)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.curve_csv)))
    out_csv = Path(args.outdir, f"{args.prefix}_curve_robust.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["s","omega2","Kphi","kappa","robust_frac"])
        for i, r in enumerate(rows):
            if i % args.sample_step: continue
            s = float(r["s"]); w2 = float(r["omega2"]); K = float(r["Kphi"]); kap = float(r["kappa"])
            rf = local_robust(w2, K, dw=args.dw, dK=args.dK, span_w=args.span_w, span_k=args.span_k,
                              steps=args.steps, si_thresh=args.si_thresh)
            w.writerow([f"{s:.6f}", f"{w2:.6f}", f"{K:.6f}", f"{kap:.6e}", f"{rf:.6f}"])
            print(f"[p19R] s={s:.3f} w2={w2:.3f} K={K:.3f}  robust={rf:.3f}")

    # simple plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, numpy as np
        R = list(csv.DictReader(open(out_csv)))
        s = np.array([float(r["s"]) for r in R]); rf = np.array([float(r["robust_frac"]) for r in R])
        fig = plt.figure(figsize=(6,3.6))
        plt.plot(s, rf, "-"); plt.xlabel("arc length s"); plt.ylabel("local robust frac (SI≥0.95)")
        fig.tight_layout(); fig.savefig(Path(args.outdir, f"{args.prefix}_robust_vs_s.png"), dpi=160); plt.close(fig)
    except Exception:
        pass

if __name__ == "__main__":
    raise SystemExit(main())
