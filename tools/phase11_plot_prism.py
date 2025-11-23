#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_rows(csvp):
    rows=[]
    with open(csvp) as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            try:
                rows.append(dict(
                    topology=r["topology"],
                    seed=int(r["seed"]),
                    alpha=float(r["alpha"]),
                    pattern=r["dw_pattern"],
                    R=float(r["R_mean"]),
                    P=float(r["pair_coh"]),
                    K=float(r["K_edge"]),
                    span=float(r.get("phase_span", 0.0)),
                    cvar=float(r.get("circ_var", 0.0)),
                    shortcuts=int(r.get("shortcuts", 0)),
                ))
            except Exception:
                pass
    return rows

def group_by_pattern(rows):
    g = {}
    for r in rows:
        g.setdefault(r["pattern"], []).append(r)
    return g

def mean_by_alpha(rs, key):
    A = sorted(rs, key=lambda x: (x["alpha"], x["seed"]))
    alphas = sorted(set([x["alpha"] for x in A]))
    y = []
    for a in alphas:
        y.append(np.mean([x[key] for x in A if x["alpha"]==a]))
    return alphas, y

def plot_by_pattern(rows, prefix="outputs/phase11/plots/prism"):
    Path("outputs/phase11/plots").mkdir(parents=True, exist_ok=True)
    byp = group_by_pattern(rows)

    # R vs alpha
    plt.figure()
    for pat, rs in byp.items():
        alphas, y = mean_by_alpha(rs, "R")
        plt.plot(alphas, y, marker='o', label=pat)
    plt.xlabel("alpha"); plt.ylabel("R_mean")
    plt.title("Phase-11: R vs alpha by pattern"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); out1 = f"{prefix}_R.png"; plt.savefig(out1, dpi=150); plt.close()

    # pair_coh vs alpha
    plt.figure()
    for pat, rs in byp.items():
        alphas, y = mean_by_alpha(rs, "P")
        plt.plot(alphas, y, marker='o', label=pat)
    plt.xlabel("alpha"); plt.ylabel("pair_coh (edge-avg cos Δθ)")
    plt.title("Phase-11: Pair coherence vs alpha"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); out2 = f"{prefix}_pair.png"; plt.savefig(out2, dpi=150); plt.close()

    # mean K vs alpha
    plt.figure()
    A = rows
    alphas = sorted(set([x["alpha"] for x in A]))
    y = [np.mean([x["K"] for x in A if x["alpha"]==a]) for a in alphas]
    plt.plot(alphas, y, marker='o')
    plt.xlabel("alpha"); plt.ylabel("mean K_edge")
    plt.title("Phase-11: mean K_edge vs alpha"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); out3 = f"{prefix}_K.png"; plt.savefig(out3, dpi=150); plt.close()

    # NEW: phase_span vs alpha (lower is better)
    plt.figure()
    for pat, rs in byp.items():
        alphas, y = mean_by_alpha(rs, "span")
        plt.plot(alphas, y, marker='o', label=pat)
    plt.xlabel("alpha"); plt.ylabel("phase_span (rad)")
    plt.title("Phase-11: Phase spread vs alpha"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); out4 = f"{prefix}_span.png"; plt.savefig(out4, dpi=150); plt.close()

    # NEW: circular variance vs alpha (lower is better)
    plt.figure()
    for pat, rs in byp.items():
        alphas, y = mean_by_alpha(rs, "cvar")
        plt.plot(alphas, y, marker='o', label=pat)
    plt.xlabel("alpha"); plt.ylabel("circular variance")
    plt.title("Phase-11: Circular variance vs alpha"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); out5 = f"{prefix}_cvar.png"; plt.savefig(out5, dpi=150); plt.close()

    print(f"[p11] wrote: {out1}, {out2}, {out3}, {out4}, {out5}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/phase11/data/prism_sweep.csv")
    ap.add_argument("--prefix", default="outputs/phase11/plots/prism")
    args = ap.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}. Run the sweep first.")
    plot_by_pattern(rows, prefix=args.prefix)
