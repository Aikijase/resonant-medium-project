#!/usr/bin/env python3
"""
Pack Phase-4 artifacts into a timestamped zip for sharing.
Includes JSONs, MD tables, TEX tables, and plots.
"""
import os, time, zipfile, pathlib

OUTDIR = "outputs/phase4"
PLOTS  = "plots"

GLOBS = [
    f"{OUTDIR}/gk_score.json",
    f"{OUTDIR}/gk_table.md",
    f"{OUTDIR}/gk_table.tex",
    f"{OUTDIR}/gk_multi.json",
    f"{OUTDIR}/gk_multi_table.md",
    f"{OUTDIR}/gk_multi_table.tex",
    f"{PLOTS}/gk_residuals.png",
    f"{PLOTS}/gk_multi_residuals.png",
]

def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    zip_path = f"{OUTDIR}/phase4_artifacts_{ts}.zip"
    pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in GLOBS:
            if os.path.exists(p):
                z.write(p)
    print(f"Wrote {zip_path}")

if __name__ == "__main__":
    main()
