#!/usr/bin/env python3
"""
Phase-5: Summarize stability sweep and damping toy runs.

Reads:
  outputs/phase5/stability_grid.json
  outputs/phase5/damping_runs.json

Writes:
  outputs/phase5/phase5_summary.md
"""
import json, os, math
from pathlib import Path

IN_GRID = "outputs/phase5/stability_grid.json"
IN_RUNS = "outputs/phase5/damping_runs.json"
OUT_MD  = "outputs/phase5/phase5_summary.md"

def main():
    Path("outputs/phase5").mkdir(parents=True, exist_ok=True)
    if not os.path.exists(IN_GRID):
        print("[warn] no stability grid JSON found; run sweep first")
        grid = None
    else:
        grid = json.load(open(IN_GRID))

    if not os.path.exists(IN_RUNS):
        print("[warn] no damping runs JSON found; run damping ODE next")
        runs = None
    else:
        runs = json.load(open(IN_RUNS))

    lines = ["# Phase-5 Summary", ""]
    if grid:
        meta = grid["meta"]
        rows = grid["grid"]
        best = min(rows, key=lambda r: r["delta_chi2"])
        lines += [
            "## Stability Sweep",
            "",
            f"- Grid: γ∈[{meta['g_range'][0]},{meta['g_range'][1]}], f∈[{meta['f_range'][0]},{meta['f_range'][1]}], A∈[{meta['A_range'][0]},{meta['A_range'][1]}]",
            f"- Nominal (γ0, f0) = ({meta['g0']:.3f}, {meta['f0']:.3f})",
            f"- Best grid point (min Δχ²): γ={best['gamma']:.3f}, f={best['f']:.3f}, A={best['A']:.3f}, Δχ²={best['delta_chi2']:.3f}",
            ""
        ]
    if runs:
        lines += ["## Damping Toy ODE", ""]
        lines += ["| label | γ | f | ζ (damping ratio) | t½ (amp) | x_max |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in runs["runs"]:
            lines.append(f"| {r['label']} | {r['gamma']:.3f} | {r['f']:.3f} | {r['zeta']:.3f} | {r['t_half']:.3f} | {r['x_max']:.3f} |")
        lines.append("")

    open(OUT_MD,"w").write("\n".join(lines))
    print(f"Wrote {OUT_MD}")

if __name__ == "__main__":
    main()
