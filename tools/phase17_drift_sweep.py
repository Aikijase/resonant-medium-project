#!/usr/bin/env python3
import argparse, json, math, os, random, subprocess, sys, csv, time
from pathlib import Path

def run_si(w2, K, steps=12000, burn=300, noise=0.01, kv=0.10, kx=0.20, eps=0.06, adapt=15, preset="neuron"):
    PY = sys.executable
    p = subprocess.run([PY, "tools/phase10_phasecouple_demo.py",
        "--preset", str(preset),
        "--omega2", str(w2),
        "--kv", str(kv), "--kx", str(kx),
        "--Kphi", str(K),
        "--eps", str(eps),
        "--adapt_every", str(adapt),
        "--noise", str(noise),
        "--steps", str(steps),
        "--burn_in", str(burn),
        "--prefix", f"p17_w{w2:.3f}_K{K:.3f}"],
        capture_output=True, text=True)
    # tolerate occasional stderr chatter; parse stdout json
    return json.loads(p.stdout)["metrics"]["sync_index"]

def load_shell_rows(path, max_rows=None, seed=1):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append((float(row["omega2"]), float(row["Kphi"]), float(row.get("sync_index", "0"))))
            except Exception:
                continue
    if max_rows and len(rows) > max_rows:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_rows]
    return rows

def sweep_anchor(anchor, cfg, outdir):
    w0, K0, si0 = anchor
    dw = cfg["grid_dw"]; dk = cfg["grid_dk"]
    half_w = cfg["span_w"]/2.0; half_k = cfg["span_k"]/2.0
    w_vals = [round(w0 + i*dw, 6) for i in range(round(-half_w/dw), round(half_w/dw)+1)]
    k_vals = [round(K0 + j*dk, 6) for j in range(round(-half_k/dk), round(half_k/dk)+1)]
    grid = []
    for w in w_vals:
        row = []
        for K in k_vals:
            si = run_si(w, K, steps=cfg["steps"], burn=cfg["burn"], noise=cfg["noise"],
                        kv=cfg["kv"], kx=cfg["kx"], eps=cfg["eps"], adapt=cfg["adapt"], preset=cfg["preset"])
            row.append(si)
        grid.append(row)
    # write CSV
    stem = f"p17_anchor_w{w0:.3f}_K{K0:.3f}"
    csv_path = Path(outdir) / f"{stem}.heatmap.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# rows=omega2 grid, cols=Kphi grid"])
        w.writerow(["omega2_grid"] + w_vals)
        w.writerow(["Kphi_grid"] + k_vals)
        for i, wv in enumerate(w_vals):
            w.writerow([wv] + grid[i])
    return {"stem": stem, "w_vals": w_vals, "k_vals": k_vals, "grid": grid, "anchor": (w0, K0, si0)}
def summarize_grid(w_vals, k_vals, grid, si_thresh):
    # robustness = fraction of cells >= threshold
    flat = [v for row in grid for v in row]
    if not flat: return {"robust_frac": 0.0, "si_mean": 0.0, "si_median": 0.0}
    flat_sorted = sorted(flat)
    mid = flat_sorted[len(flat)//2]
    robust = sum(1 for v in flat if v >= si_thresh) / len(flat)
    return {"robust_frac": robust, "si_mean": sum(flat)/len(flat), "si_median": mid}

def save_summary_row(fw, anchor, stem, summ, cfg):
    w0, K0, si0 = anchor
    fw.writerow({
        "omega2_anchor": f"{w0:.6f}",
        "Kphi_anchor": f"{K0:.6f}",
        "si_anchor": f"{si0:.6f}",
        "stem": stem,
        "grid_dw": cfg["grid_dw"],
        "grid_dk": cfg["grid_dk"],
        "span_w": cfg["span_w"],
        "span_k": cfg["span_k"],
        "steps": cfg["steps"],
        "si_thresh": cfg["si_thresh"],
        "robust_frac": f"{summ['robust_frac']:.6f}",
        "si_mean": f"{summ['si_mean']:.6f}",
        "si_median": f"{summ['si_median']:.6f}",
    })

def plot_heatmap_png(outdir, stem, w_vals, k_vals, grid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        arr = np.array(grid, dtype=float)
        fig = plt.figure(figsize=(6,5))
        im = plt.imshow(arr, origin="lower", aspect="auto",
                        extent=[min(k_vals), max(k_vals), min(w_vals), max(w_vals)])
        plt.colorbar(im, label="sync_index")
        plt.xlabel("Kphi")
        plt.ylabel("omega2")
        plt.title(stem)
        fig.tight_layout()
        png = Path(outdir) / f"{stem}.heatmap.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
    except Exception as e:
        sys.stderr.write(f"[warn] matplotlib plot failed: {e}\n")

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def main():
    ap = argparse.ArgumentParser(description="Phase-17 Drift Robustness Sweep")
    ap.add_argument("--shell-csv", required=True, help="Phase-14 shell CSV (with columns omega2,Kphi,sync_index)")
    ap.add_argument("--outdir", default="outputs/phase17")
    ap.add_argument("--prefix", default="p17")
    ap.add_argument("--sample", type=int, default=12, help="Number of anchor points to sample")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid-dw", type=float, default=0.02)
    ap.add_argument("--grid-dk", type=float, default=0.01)
    ap.add_argument("--span-w", type=float, default=0.12)
    ap.add_argument("--span-k", type=float, default=0.06)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--burn", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--adapt", type=int, default=15)
    ap.add_argument("--preset", default="neuron")
    ap.add_argument("--si-thresh", type=float, default=0.95)
    args = ap.parse_args()

    cfg = vars(args).copy()
    ensure_dir(args.outdir)
    anchors = load_shell_rows(args.shell_csv, max_rows=args.sample, seed=args.seed)
    if not anchors:
        print("no anchors found in shell csv", file=sys.stderr); return 2

    # summary file
    summary_csv = Path(args.outdir) / f"{args.prefix}_drift_summary.csv"
    with open(summary_csv, "w", newline="") as fsum:
        fw = csv.DictWriter(fsum, fieldnames=[
            "omega2_anchor","Kphi_anchor","si_anchor","stem",
            "grid_dw","grid_dk","span_w","span_k","steps","si_thresh",
            "robust_frac","si_mean","si_median"
        ])
        fw.writeheader()

        for idx, anc in enumerate(anchors, 1):
            t0 = time.time()
            res = sweep_anchor(anc, cfg, args.outdir)
            summ = summarize_grid(res["w_vals"], res["k_vals"], res["grid"], args.si_thresh)
            save_summary_row(fw, res["anchor"], res["stem"], summ, cfg)
            plot_heatmap_png(args.outdir, res["stem"], res["w_vals"], res["k_vals"], res["grid"])
            dt = time.time() - t0
            print(f"[p17] anchor {idx}/{len(anchors)}  w2={anc[0]:.3f} Kphi={anc[1]:.3f}  robust={summ['robust_frac']:.3f}  ({dt:.1f}s)")

    # tiny report
    with open(Path(args.outdir)/f"{args.prefix}_report.md", "w") as f:
        f.write(f"# Phase 17 — Drift Robustness Sweep\n")
        f.write(f"- shell_csv: {args.shell_csv}\n")
        f.write(f"- sample: {args.sample}  grid: dω={args.grid_dw} span={args.span_w} ; dK={args.grid_dk} span={args.span_k}\n")
        f.write(f"- steps: {args.steps} burn: {args.burn} preset: {args.preset} noise: {args.noise}\n")
        f.write(f"- si_thresh: {args.si_thresh}\n")
        f.write(f"- summary_csv: {summary_csv}\n")

if __name__ == "__main__":
    sys.exit(main())
