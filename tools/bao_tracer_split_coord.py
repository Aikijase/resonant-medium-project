#!/usr/bin/env python3
# BAO tracer split: coordinate descent and grid scan with live progress, ETA, and resume
import argparse, sys, time, re, math, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ENGINE = Path("tools/bao_tracer_sweep.py")

# -------------------------- Utils --------------------------
def sh_stream(cmd, env=None):
    """Run a command, streaming stdout; return combined text (raises on nonzero)."""
    print("[run]", " ".join(map(str, cmd)))
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        out_lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError("Command failed: " + " ".join(map(str, cmd)))
    return "".join(out_lines)

def parse_best(txt):
    """
    Parse 'Best: hELG=...  hLRG2=...  chi2=...  dof=...  chi2/dof=...'
    from bao_tracer_sweep.py output.
    """
    m = re.search(r"Best:\s*hELG=([0-9.]+)\s+hLRG2=([0-9.]+)\s+chi2=([0-9.]+)\s+dof=([0-9]+)\s+chi2/dof=([0-9.]+)", txt)
    if not m:
        # Fallback: just chi2
        m2 = re.search(r"chi2=([0-9.]+)", txt)
        if m2:
            return {"chi2": float(m2.group(1))}
        raise ValueError("Could not parse 'Best:' line from engine output.")
    return {
        "hELG": float(m.group(1)),
        "hLRG2": float(m.group(2)),
        "chi2": float(m.group(3)),
        "dof": int(m.group(4)),
        "chi2_dof": float(m.group(5)),
    }

def fmt_eta(seconds):
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return "ETA --:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"ETA {h:d}h {m:02d}m"
    return f"ETA {m:02d}m {s:02d}s"

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

# -------------------------- Engine call --------------------------
def run_single(a, df_csv, hE, hL, out_prefix):
    """
    Call the sweep engine at a single point by giving degenerate ranges (1 step).
    """
    cmd = [
        sys.executable, str(ENGINE),
        "--bao-csv", str(df_csv),
        "--cov-csv", str(a.cov_csv),
        "--om0", str(a.om0),
        "--h", str(a.h),
        "--z-col", a.z_col, "--y-col", a.y_col, "--kind-col", a.kind_col,
        "--label-col", a.label_col,
        "--elg-pattern", a.elg_pattern, "--lrg2-pattern", a.lrg2_pattern,
        "--hELG-range", f"{hE}", f"{hE}", "1",
        "--hLRG2-range", f"{hL}", f"{hL}", "1",
        "--out-prefix", f"{out_prefix}_hE{hE:.6f}_hL{hL:.6f}",
    ]
    out = sh_stream(cmd)
    best = parse_best(out)
    # ensure chi2 exists
    chi2 = float(best.get("chi2"))
    return chi2, best

# -------------------------- Grid mode --------------------------
def run_grid(a, df, hE_grid, hL_grid):
    """
    Grid scan with:
    - incremental CSV append (resume-safe),
    - live progress + ETA,
    - streaming engine output per point.
    """
    out_csv = Path(f"{a.out_prefix}_heatmap_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resume: read existing points
    done = set()
    if out_csv.exists():
        try:
            old = pd.read_csv(out_csv)
            for _, r in old.iterrows():
                done.add((round(float(r["hELG"]), 6), round(float(r["hLRG2"]), 6)))
        except Exception:
            pass
    else:
        out_csv.write_text("hELG,hLRG2,chi2\n")

    # Write a filtered/labeled CSV (already labeled if label_col is present)
    tmp = Path("/tmp/bao_split_tmp.csv")
    df.to_csv(tmp, index=False)

    total = len(hE_grid) * len(hL_grid)
    already = len(done)
    start = time.time()

    with open(out_csv, "a") as f:
        count = already
        for i, hE in enumerate(hE_grid):
            for j, hL in enumerate(hL_grid):
                key = (round(float(hE), 6), round(float(hL), 6))
                if key in done:
                    count += 1
                    # quick status tick
                    elapsed = time.time() - start
                    rate = (count - already) / elapsed if elapsed > 0 else 0.0
                    remaining = max(total - count, 0)
                    eta = remaining / rate if rate > 0 else None
                    print(f"[skip] {key}  {count}/{total}  {rate:.2f} pts/s  {fmt_eta(eta)}")
                    continue

                # Run one point
                try:
                    chi2, _best = run_single(a, tmp, hE, hL, a.out_prefix)
                except Exception as e:
                    print(f"[warn] point failed hELG={hE:.6f}, hLRG2={hL:.6f}: {e}")
                    chi2 = float("nan")

                # Append immediately
                f.write(f"{hE:.6f},{hL:.6f},{chi2:.6f}\n")
                f.flush()

                # Progress
                count += 1
                elapsed = time.time() - start
                rate = (count - already) / elapsed if elapsed > 0 else 0.0
                remaining = max(total - count, 0)
                eta = remaining / rate if rate > 0 else None
                pct = 100.0 * count / total
                bar_n = 24
                filled = int(bar_n * pct / 100.0)
                bar = "█" * filled + "·" * (bar_n - filled)
                print(f"[{bar}] {count}/{total} ({pct:5.1f}%)  rate={rate:.2f} pts/s  {fmt_eta(eta)}  chi2={chi2:.3f}")

    grid = pd.read_csv(out_csv)
    # quick best readout
    k = grid["chi2"].idxmin()
    br = grid.loc[k]
    print(f"[done {now_iso()}] BEST grid point: hELG={br.hELG:.6f}  hLRG2={br.hLRG2:.6f}  chi2={br.chi2:.3f}")
    return grid

# -------------------------- Coord descent (optional) --------------------------
def coord_descent(a, df, hE0, hL0, n_iter=6, n_steps=61, span=0.06):
    """
    Alternate sweeping hELG|hLRG2 lines; writes coord trace + per-sweep CSVs.
    """
    tmp = Path("/tmp/bao_split_tmp.csv")
    df.to_csv(tmp, index=False)

    path = []
    hE, hL = float(hE0), float(hL0)
    for r in range(n_iter):
        # Sweep ELG around hE
        hE_grid = np.linspace(hE - span, hE + span, n_steps)
        outE = Path(f"{a.out_prefix}_iter{r}_sweepELG_summary.csv")
        with open(outE, "w") as g:
            g.write("hELG,chi2\n")
            for x in hE_grid:
                chi2, _ = run_single(a, tmp, x, hL, a.out_prefix)
                g.write(f"{x:.6f},{chi2:.6f}\n")
        dE = pd.read_csv(outE)
        hE = float(dE.loc[dE["chi2"].idxmin(), "hELG"])

        # Sweep LRG2 around hL
        hL_grid = np.linspace(hL - span, hL + span, n_steps)
        outL = Path(f"{a.out_prefix}_iter{r}_sweepLRG2_summary.csv")
        with open(outL, "w") as g:
            g.write("hLRG2,chi2\n")
            for y in hL_grid:
                chi2, _ = run_single(a, tmp, hE, y, a.out_prefix)
                g.write(f"{y:.6f},{chi2:.6f}\n")
        dL = pd.read_csv(outL)
        hL = float(dL.loc[dL["chi2"].idxmin(), "hLRG2"])

        # Trace
        path.append((r + 1, hE, hL, min(dE["chi2"].min(), dL["chi2"].min())))
        print(f"[iter {r+1}] hELG={hE:.6f}  hLRG2={hL:.6f}  chi2~{path[-1][3]:.3f}")

    # final best eval
    chi2, _ = run_single(a, tmp, hE, hL, a.out_prefix)
    Path(f"{a.out_prefix}_split_best.json").write_text(json.dumps({"hELG": hE, "hLRG2": hL, "chi2": chi2}, indent=2))
    with open(f"{a.out_prefix}_coord_trace.csv", "w") as g:
        g.write("iter,hELG,hLRG2,chi2\n")
        for it, x, y, c in path:
            g.write(f"{it},{x:.6f},{y:.6f},{c:.6f}\n")
    print(f"[coord] BEST: hELG={hE:.6f}  hLRG2={hL:.6f}  chi2={chi2:.3f}")
    return hE, hL, chi2

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bao-csv", required=True)
    ap.add_argument("--cov-csv", required=True)
    ap.add_argument("--om0", type=float, required=True)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--z-col", default="z")
    ap.add_argument("--y-col", default="y_data")
    ap.add_argument("--kind-col", default="kind")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--elg-pattern", default="ELG")
    ap.add_argument("--lrg2-pattern", default="LRG2")
    ap.add_argument("--out-prefix", required=True)

    # modes
    ap.add_argument("--grid", action="store_true", help="Run full grid sweep")
    ap.add_argument("--hELG-range", nargs=3, type=float, metavar=("MIN","MAX","STEPS"))
    ap.add_argument("--hLRG2-range", nargs=3, type=float, metavar=("MIN","MAX","STEPS"))

    ap.add_argument("--coord", action="store_true", help="Coordinate descent mode")
    ap.add_argument("--start", nargs=2, type=float, metavar=("hELG0","hLRG20"))

    a = ap.parse_args()

    # load BAO CSV (already labeled)
    df = pd.read_csv(a.bao_csv)

    if a.grid:
        if a.hELG_range is None or a.hLRG2_range is None:
            raise SystemExit("--grid requires both --hELG-range and --hLRG2-range")
        hE_min, hE_max, nE = a.hELG_range
        hL_min, hL_max, nL = a.hLRG2_range
        hE_grid = np.linspace(hE_min, hE_max, int(nE))
        hL_grid = np.linspace(hL_min, hL_max, int(nL))
        run_grid(a, df, hE_grid, hL_grid)
        return

    if a.coord:
        if a.start is None:
            raise SystemExit("--coord requires --start hELG0 hLRG20")
        coord_descent(a, df, a.start[0], a.start[1])
        return

    raise SystemExit("Select a mode: --grid or --coord")

if __name__ == "__main__":
    main()
