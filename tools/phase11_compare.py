#!/usr/bin/env python3
import csv, sys, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

def load_csv(p, pattern_filter="block"):
    rows=[]
    with open(p) as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            if r["dw_pattern"] != pattern_filter: 
                continue
            rows.append(dict(
                alpha=float(r["alpha"]),
                seed=int(r["seed"]),
                R=float(r["R_mean"]),
                P=float(r["pair_coh"]),
                K=float(r["K_edge"]),
                span=float(r.get("phase_span",0.0)),
                cvar=float(r.get("circ_var",0.0)),
                shortcuts=int(r.get("shortcuts",0)),
            ))
    return rows

def mean_by_alpha(rows, key):
    bucket=defaultdict(list)
    for r in rows:
        bucket[r["alpha"]].append(r[key])
    alphas=sorted(bucket)
    vals=[np.mean(bucket[a]) for a in alphas]
    return alphas, vals

def first_alpha_meeting(rows, target):
    # returns alpha* where mean R >= target
    alphas, R = mean_by_alpha(rows, "R")
    for a, r in zip(alphas, R):
        if r >= target: 
            return a, r
    return None, None

def main():
    if len(sys.argv) < 4:
        print("usage: phase11_compare.py <base_csv> <short_csv> <out_prefix>")
        sys.exit(1)
    base_csv, short_csv, out_prefix = sys.argv[1:4]
    base = load_csv(base_csv)
    short = load_csv(short_csv)

    ab, Rb = mean_by_alpha(base, "R")
    as_, Rs = mean_by_alpha(short, "R")
    _, Sb = mean_by_alpha(base, "span")
    _, Ss = mean_by_alpha(short, "span")
    _, Kb = mean_by_alpha(base, "K")
    _, Ks = mean_by_alpha(short, "K")

    # R overlay
    plt.figure()
    plt.plot(ab, Rb, marker='o', label='chain (no shortcut)')
    plt.plot(as_, Rs, marker='o', label='chain + 1 shortcut')
    plt.xlabel("alpha"); plt.ylabel("R_mean")
    plt.title("Phase-11: Global coherence (block pattern)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    p1 = f"{out_prefix}_R_overlay.png"; plt.savefig(p1, dpi=150); plt.close()

    # span overlay
    plt.figure()
    plt.plot(ab, Sb, marker='o', label='chain (no shortcut)')
    plt.plot(as_, Ss, marker='o', label='chain + 1 shortcut')
    plt.xlabel("alpha"); plt.ylabel("phase_span (rad)  ↓ better")
    plt.title("Phase-11: Phase spread (block pattern)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    p2 = f"{out_prefix}_span_overlay.png"; plt.savefig(p2, dpi=150); plt.close()

    # Print a small summary table
    print("alpha, R_base, R_short, dR, span_base, span_short, dspan, K_base, K_short")
    # join by alphas present in both sets
    A = sorted(set(ab) & set(as_))
    for i, a in enumerate(A):
        ib = ab.index(a); is_ = as_.index(a)
        dR = Rs[is_] - Rb[ib]
        dS = Ss[is_] - Sb[ib]
        print(f"{a:.2f}, {Rb[ib]:.3f}, {Rs[is_]:.3f}, {dR:+.3f}, {Sb[ib]:.3f}, {Ss[is_]:.3f}, {dS:+.3f}, {Kb[ib]:.3f}, {Ks[is_]:.3f}")

    # thresholds
    for thr in (0.90, 0.95, 0.98):
        ab_star, Rb_star = first_alpha_meeting(base, thr)
        as_star, Rs_star = first_alpha_meeting(short, thr)
        print(f"alpha* for R≥{thr:.2f}: base={ab_star}, short1={as_star}")

    print(f"[p11] wrote: {p1}, {p2}")

if __name__ == "__main__":
    main()
