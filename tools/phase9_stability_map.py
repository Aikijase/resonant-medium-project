#!/usr/bin/env python3
"""
Phase 9 — Stability Map (β × damping)
Builds a verdict map over (beta, damping) showing regions:
  - "Near-resonant" (large amplitude, low precision)
  - "Stable/over-damped" (low amplitude, high precision)
  - "Adaptive window" (balanced)
Heuristic is consistent with Phase-9 report_card and beta sweep behavior.

Usage:
  env PYTHONPATH=. python3 tools/phase9_stability_map.py \
    --inputs outputs/phase9/physical_run.csv outputs/phase9/psychological_run_noscipy.csv outputs/phase9/learning_run_noscipy.csv \
    --beta 0.02 0.80 60 \
    --damping 0.00 0.80 60 \
    --out-prefix outputs/phase9/stability
"""
import argparse, os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt

def analyze(df):
    vals = df.select_dtypes("number").values.flatten()
    vals = vals[~np.isnan(vals)]
    amp = np.ptp(vals)
    err = np.sqrt(np.mean((vals - np.mean(vals))**2))
    # tempo proxy
    beta0 = float(np.mean(np.gradient(vals)))
    return amp, err, beta0

def verdict(amplitude, error):
    # conservative thresholds scaled to units ~ your Phase 9 table
    if amplitude >= 8 and error >= 2.0:
        return 2  # Near-resonant
    if amplitude <= 0.2 and error <= 0.02:
        return 0  # Stable/over-damped
    return 1      # Adaptive

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--beta", nargs=3, type=float, metavar=("BMIN","BMAX","NSTEPS"), required=True)
    ap.add_argument("--damping", nargs=3, type=float, metavar=("DMIN","DMAX","NSTEPS"), required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # baseline stats (use physical as amplitude anchor, psych as stability anchor)
    dfs = [pd.read_csv(p) for p in args.inputs]
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.inputs]
    stats = {n: analyze(d) for n,d in zip(names, dfs)}
    # pick anchors if present
    amp_phys, err_phys, beta_phys = stats.get("physical_run", stats[names[0]])
    amp_psych, err_psych, beta_psych = stats.get("psychological_run_noscipy", stats[names[-1]])

    bmin,bmax,bN = args.beta
    dmin,dmax,dN = args.damping
    betas = np.linspace(bmin,bmax,int(bN))
    damps = np.linspace(dmin,dmax,int(dN))

    Z = np.zeros((len(damps), len(betas)), dtype=int)
    AMP = np.zeros_like(Z, dtype=float)
    ERR = np.zeros_like(Z, dtype=float)

    # simple paramization:
    # amplitude decreases with beta and damping; error grows with beta but shrinks with damping
    for i,dm in enumerate(damps):
        for j,be in enumerate(betas):
            # scale between phys and psych anchors
            amp = amp_phys*(1 - 0.55*be)*(1 - 0.65*dm) + amp_psych*(0.15*dm)
            amp = max(amp, 0.0)
            err = err_phys*(1 + 1.8*be)*(1 - 0.6*dm) + err_psych*(0.4*dm)
            AMP[i,j] = amp
            ERR[i,j] = err
            Z[i,j] = verdict(amp, err)

    # plot verdict map
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(Z, origin="lower",
                   extent=[betas[0], betas[-1], damps[0], damps[-1]],
                   aspect="auto", interpolation="nearest")
    # colorbar with custom ticks
    cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2])
    cbar.ax.set_yticklabels(["Stable", "Adaptive", "Near-resonant"])
    ax.set_xlabel("β (tempo gain)")
    ax.set_ylabel("Damping")
    ax.set_title("Phase 9 — Stability Regions (β × damping)")
    plt.tight_layout()
    map_png = args.out_prefix + "_map.png"
    plt.savefig(map_png, dpi=200)
    plt.close(fig)

    # frontier overlay (optional context)
    fig2 = plt.figure(figsize=(7,5))
    for n,(a,e,_) in stats.items():
        plt.scatter(e, a, label=n)
    plt.xlabel("RMS Error"); plt.ylabel("Peak-to-Peak Amplitude")
    plt.title("Anchors — Amplitude vs Error")
    plt.grid(True, alpha=0.4); plt.legend()
    front_png = args.out_prefix + "_anchors.png"
    plt.tight_layout(); plt.savefig(front_png, dpi=200); plt.close(fig2)

    # export CSVs
    # grid long-form
    rows = []
    for i,dm in enumerate(damps):
        for j,be in enumerate(betas):
            rows.append(dict(beta=be, damping=dm, amplitude=AMP[i,j], error=ERR[i,j], label=Z[i,j]))
    grid_csv = args.out_prefix + "_grid.csv"
    pd.DataFrame(rows).to_csv(grid_csv, index=False)

    json.dump({
        "inputs":[os.path.basename(p) for p in args.inputs],
        "map_png": os.path.basename(map_png),
        "anchors_png": os.path.basename(front_png),
        "grid_csv": os.path.basename(grid_csv),
        "labels": { "0":"Stable/over-damped", "1":"Adaptive", "2":"Near-resonant" }
    }, open(args.out_prefix + "_manifest.json","w"), indent=2)

    print(f"[phase9] Saved: {map_png}")
    print(f"[phase9] Saved: {front_png}")
    print(f"[phase9] Saved: {grid_csv}")
    print(f"[phase9] Saved: {args.out_prefix + '_manifest.json'}")

if __name__ == "__main__":
    main()
