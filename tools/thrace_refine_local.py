#!/usr/bin/env python3
"""
Refine locally around best (alpha, gamma) from multistart, holding beta_* fixed.

Reads:
  outputs/thrace_multistart/analysis/best_params.json

Does:
  - Builds a log-space grid around alpha,gamma (default: ±0.5 dex, 9×9 grid)
  - Runs thrace_chain for each point (using best beta_de, beta_dm)
  - Skips failures, logs each run to outputs/thrace_refine/logs/run_aXX_gYY.log
  - Writes results to outputs/thrace_refine/leaderboard.csv and best_row.json
  - Plots a χ² heatmap if any successes
  - If < 10% success, auto-shrinks spans by 50% and retries once

Usage:
  PYTHONPATH=. python tools/thrace_refine_local.py \
    --rho-bh data/rho_bh_z.clean.csv \
    --rho-bh-col rho_bh_Msun_per_Mpc3 \
    --config configs/joint_baseline.json \
    --grid 9 \
    --alpha-span 0.5 \
    --gamma-span 0.5 \
    --tag-prefix refine
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

REFINE_DIR = ROOT / "outputs" / "thrace_refine"
LOGS_DIR   = REFINE_DIR / "logs"
LB_CSV     = REFINE_DIR / "leaderboard.csv"

def _load_best() -> Dict[str, float]:
    p = ROOT / "outputs/thrace_multistart/analysis/best_params.json"
    if not p.exists():
        raise SystemExit(f"[refine] missing {p} (run thrace_ms_analyze first)")
    d = json.loads(p.read_text())
    for k in ["alpha","gamma","beta_de","beta_dm"]:
        if k not in d or d[k] is None:
            raise SystemExit(f"[refine] best_params.json missing key: {k}")
    return d

def _logspace(center: float, span_dex: float, n: int) -> List[float]:
    lo = math.log10(center) - span_dex
    hi = math.log10(center) + span_dex
    if n <= 1:
        return [center]
    return [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]

def _run_and_tee(cmd: List[str], log_path: Path) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[refine] RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    with log_path.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        if proc.stdout:
            f.write("\n[stdout]\n" + proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n" + proc.stderr)
        f.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode == 0:
        print(f"[refine] ok → {log_path}", flush=True)
    else:
        print(f"[refine] FAIL(code={proc.returncode}) → {log_path}", flush=True)
    return proc

def _parse_run_dir(stdout: str) -> Optional[str]:
    for line in (stdout or "").splitlines():
        s = line.strip()
        if s.startswith("run_dir") or s.startswith("run_dir") or s.startswith("run_dir"):
            parts = line.split(":", 1)
            if len(parts) >= 2:
                return parts[1].strip()
        if s.startswith("run_dir") or s.startswith("run_dir:"):
            parts = line.split(":", 1)
            if len(parts) >= 2:
                return parts[1].strip()
        if s.startswith("run_dir") or s.startswith("run_dir          :"):
            parts = line.split(":", 1)
            if len(parts) >= 2:
                return parts[1].strip()
        if s.startswith("run_dir") or s.startswith(" run_dir"):
            parts = line.split(":", 1)
            if len(parts) >= 2:
                return parts[1].strip()
    return None

def _try_load_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    summary_path = ROOT / run_dir / "joint.std_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text())
    except Exception:
        return None

def _chi2_from_summary(std: Dict[str, Any]) -> Optional[float]:
    v = std.get("chi2")
    return float(v) if isinstance(v, (int, float)) else None

def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["i","j","alpha","gamma","beta_de","beta_dm","chi2","AIC","BIC","dChi2","run_dir","summary_path","log"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})

def _grid_refine(rho_bh: str, rho_bh_col: str, z_col: Optional[str], config: str,
                 tag_prefix: str, grid: int, alpha_span: float, gamma_span: float) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    best = _load_best()
    alpha0, gamma0 = float(best["alpha"]), float(best["gamma"])
    beta_de, beta_dm = float(best["beta_de"]), float(best["beta_dm"])

    A = _logspace(alpha0, alpha_span, grid)
    G = _logspace(gamma0, gamma_span, grid)

    successes, failures = 0, 0
    best_row: Optional[Dict[str, Any]] = None

    for i, a in enumerate(A):
        for j, g in enumerate(G):
            tag = f"{tag_prefix}_a{i:02d}_g{j:02d}"
            cmd = [
                str(PY), str(ROOT / "tools" / "thrace_chain.py"),
                "--rho-bh", rho_bh, "--rho-bh-col", rho_bh_col,
                "--config", config, "--tag", tag,
                "--alpha", f"{a}", "--gamma", f"{g}",
                "--beta-de", f"{beta_de}", "--beta-dm", f"{beta_dm}",
            ]
            if z_col:
                cmd += ["--z-col", z_col]

            log_path = LOGS_DIR / f"run_a{i:02d}_g{j:02d}.log"
            proc = _run_and_tee(cmd, log_path)

            if proc.returncode != 0:
                failures += 1
                _append_row(LB_CSV, {
                    "i": i, "j": j, "alpha": a, "gamma": g,
                    "beta_de": beta_de, "beta_dm": beta_dm,
                    "chi2": None, "AIC": None, "BIC": None, "dChi2": None,
                    "run_dir": None, "summary_path": None, "log": str(log_path.relative_to(ROOT)),
                })
                continue

            run_dir = _parse_run_dir(proc.stdout)
            if not run_dir:
                failures += 1
                _append_row(LB_CSV, {
                    "i": i, "j": j, "alpha": a, "gamma": g,
                    "beta_de": beta_de, "beta_dm": beta_dm,
                    "chi2": None, "AIC": None, "BIC": None, "dChi2": None,
                    "run_dir": None, "summary_path": None, "log": str(log_path.relative_to(ROOT)),
                })
                print(f"[refine] WARN: could not parse run_dir for a={a}, g={g}")
                continue

            std = _try_load_summary(run_dir)
            if not std:
                failures += 1
                _append_row(LB_CSV, {
                    "i": i, "j": j, "alpha": a, "gamma": g,
                    "beta_de": beta_de, "beta_dm": beta_dm,
                    "chi2": None, "AIC": None, "BIC": None, "dChi2": None,
                    "run_dir": run_dir, "summary_path": f"{run_dir}/joint.std_summary.json",
                    "log": str(log_path.relative_to(ROOT)),
                })
                print(f"[refine] WARN: summary missing/invalid for {run_dir}")
                continue

            chi2 = _chi2_from_summary(std)
            AIC  = std.get("AIC")
            BIC  = std.get("BIC")
            dChi2= std.get("dChi2")

            if chi2 is None:
                failures += 1
                print(f"[refine] WARN: χ² is null at a={a}, g={g} (dir={run_dir})")
            else:
                successes += 1
                if best_row is None or chi2 < best_row["chi2"]:
                    best_row = {
                        "i": i, "j": j, "alpha": a, "gamma": g,
                        "beta_de": beta_de, "beta_dm": beta_dm,
                        "chi2": chi2, "AIC": AIC, "BIC": BIC, "dChi2": dChi2,
                        "run_dir": run_dir, "summary_path": f"{run_dir}/joint.std_summary.json",
                        "log": str(log_path.relative_to(ROOT)),
                    }

            _append_row(LB_CSV, {
                "i": i, "j": j, "alpha": a, "gamma": g,
                "beta_de": beta_de, "beta_dm": beta_dm,
                "chi2": chi2, "AIC": AIC, "BIC": BIC, "dChi2": dChi2,
                "run_dir": run_dir, "summary_path": f"{run_dir}/joint.std_summary.json" if run_dir else None,
                "log": str(log_path.relative_to(ROOT)),
            })

    return successes, failures, best_row

def _maybe_heatmap():
    try:
        import numpy as np, matplotlib.pyplot as plt
        # Load the CSV into a grid of chi2 where i,j exist and chi2 is numeric
        with LB_CSV.open() as f:
            rd = list(csv.DictReader(f))
        if not rd:
            print("[refine] (heatmap) no rows")
            return
        # collect i, j, chi2
        points = []
        for r in rd:
            try:
                i = int(float(r["i"])); j = int(float(r["j"]))
                chi2 = float(r["chi2"]) if r["chi2"] not in (None, "", "None") else float("nan")
                points.append((i, j, chi2))
            except Exception:
                continue
        if not points:
            print("[refine] (heatmap) no numeric χ²")
            return
        I = max(p[0] for p in points) + 1
        J = max(p[1] for p in points) + 1
        Z = np.full((I, J), np.nan)
        for (i, j, z) in points:
            if 0 <= i < I and 0 <= j < J:
                Z[i, j] = z
        fig = plt.figure(figsize=(6,5))
        im = plt.imshow(Z, origin="lower", aspect="auto")
        plt.xlabel("j (gamma index)"); plt.ylabel("i (alpha index)")
        plt.title("χ² over local grid (indices)")
        cbar = plt.colorbar(im); cbar.set_label("χ²")
        out = REFINE_DIR / "heatmap_local_alpha_gamma.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[refine] wrote {out}")
    except Exception as e:
        print(f"[refine] (plot skipped): {e}")

def main():
    ap = argparse.ArgumentParser(description="Local grid refine around best (alpha,gamma).")
    ap.add_argument("--rho-bh", required=True)
    ap.add_argument("--rho-bh-col", required=True)
    ap.add_argument("--z-col", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag-prefix", default="refine")
    ap.add_argument("--grid", type=int, default=9, help="grid size per axis (odd recommended)")
    ap.add_argument("--alpha-span", type=float, default=0.5, help="±dex span around best alpha")
    ap.add_argument("--gamma-span", type=float, default=0.5, help="±dex span around best gamma")
    ap.add_argument("--auto-shrink", action="store_true", default=True,
                    help="auto-shrink spans by 50% and retry once if success<10%")
    args = ap.parse_args()

    REFINE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("[refine] ===============================================")
    print(f"[refine] grid={args.grid}  alpha-span=±{args.alpha_span} dex  gamma-span=±{args.gamma_span} dex")
    print(f"[refine] tag-prefix={args.tag_prefix}")
    print("[refine] ===============================================")

    # attempt 1
    ok, bad, best_row = _grid_refine(
        rho_bh=args.rho_bh, rho_bh_col=args.rho_bh_col, z_col=args.z_col,
        config=args.config, tag_prefix=args.tag_prefix,
        grid=args.grid, alpha_span=args.alpha_span, gamma_span=args.gamma_span
    )
    total = ok + bad
    print(f"[refine] pass#1: success={ok}, fail={bad}, total={total}")

    # auto shrink if almost all failed
    if args.auto_shrink and total > 0 and ok / max(1, total) < 0.10:
        a2 = max(0.1, args.alpha_span * 0.5)
        g2 = max(0.1, args.gamma_span * 0.5)
        print(f"[refine] low success rate; retrying with tighter spans: alpha-span=±{a2}, gamma-span=±{g2}")
        ok2, bad2, best_row2 = _grid_refine(
            rho_bh=args.rho_bh, rho_bh_col=args.rho_bh_col, z_col=args.z_col,
            config=args.config, tag_prefix=args.tag_prefix + "_tight",
            grid=args.grid, alpha_span=a2, gamma_span=g2
        )
        ok += ok2; bad += bad2
        if (best_row2 and (not best_row or best_row2["chi2"] < best_row["chi2"])):
            best_row = best_row2
        print(f"[refine] pass#2: success={ok2}, fail={bad2} (cumulative ok={ok}, fail={bad})")

    # write best
    if best_row:
        (REFINE_DIR / "best_row.json").write_text(json.dumps(best_row, indent=2))
        print(f"[refine] BEST χ²={best_row['chi2']} at alpha={best_row['alpha']} gamma={best_row['gamma']}")
    else:
        print("[refine] no successful runs captured")

    # heatmap from captured rows (index heatmap – independent of physical scales)
    _maybe_heatmap()

    print(f"[refine] DONE — captured {ok} successes, {bad} failures")
    print(f"[refine] leaderboard: {LB_CSV}")
    print(f"[refine] logs: {LOGS_DIR}")

if __name__ == "__main__":
    main()
