#!/usr/bin/env python3
import os, json, subprocess, shutil, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs" / "phase2"
PLOT = ROOT / "plots"
REPORT = OUT / "REPORT.md"

def run_py(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def maybe_run_py(path, args=None, must=False):
    p = ROOT / path
    if p.exists():
        run_py(["python3", str(p), *(args or [])])
    elif must:
        raise SystemExit(f"Missing required script: {path}")
    else:
        print(f"(skip) {path} not found")

def read_json(path):
    p = ROOT / path
    if not p.exists(): return None
    try:
        return json.loads((p).read_text())
    except Exception:
        return None

def ensure_report():
    OUT.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        REPORT.write_text("# Phase-2 Report\n")
    return REPORT.read_text()

def fmt(num, nd=3):
    try:
        return f"{float(num):.{nd}f}"
    except Exception:
        return str(num)

def append_fs8_section(buf):
    Fs = read_json("outputs/phase2/fs8_eval.json")
    if not Fs: return buf
    chi2 = Fs.get("chi2"); n = Fs.get("n"); k = Fs.get("k")
    AIC = Fs.get("AIC",{}).get("value"); dA = Fs.get("AIC",{}).get("delta")
    BIC = Fs.get("BIC",{}).get("value"); dB = Fs.get("BIC",{}).get("delta")
    WWI = Fs.get("WWI")
    alpha = Fs.get("alpha")
    buf += (
        "\n## fσ8 growth (Phase-2)\n"
        f"- χ²: **{fmt(chi2)}** (n={n}, k={k})\n"
        f"- AIC/BIC: **{fmt(AIC)} / {fmt(BIC)}**, ΔAIC/ΔBIC: **{fmt(dA)} / {fmt(dB)}**\n"
        f"- WWI: **{WWI}**\n"
        f"- Amplitude fit α (S8-like): **{fmt(alpha)}**\n"
        f"- Plot: `plots/fs8_overlay.png`, residuals: `plots/fs8_residuals.png`\n"
    )
    return buf

def append_isw_amp_section(buf):
    Isw = read_json("outputs/phase2/isw_score.json")
    if not Isw: return buf
    buf += (
        "\n## ISW amplitude (WISE + RACS)\n"
        f"- Combined χ²: **{fmt(Isw.get('chi2'))}** (n={Isw.get('n')}, k={Isw.get('k')})\n"
        f"- AIC: **{fmt(Isw.get('AIC',{}).get('value'))}**, ΔAIC vs LCDM(A=1): **{fmt(Isw.get('AIC',{}).get('delta'))}**\n"
        f"- WWI: **{Isw.get('WWI')}**\n\n"
        "| survey | z_eff | A_model | A_obs ± σ | resid |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for r in Isw.get("rows", []):
        buf += f"| {r['survey']} | {fmt(r['z_eff'],2)} | {fmt(r['A_model'])} | {fmt(r['A_ISW'])} ± {fmt(r['sigma_A'])} | {fmt(r['resid'])} |\n"
    if (PLOT/"isw_amplitude.png").exists():
        buf += "\n- Figure: `plots/isw_amplitude.png`\n"
    return buf

def append_stability_sections(buf):
    has_heat = (PLOT/"stability_heatmap.png").exists()
    has_basin = (PLOT/"resonant_basins.png").exists()
    if not (has_heat or has_basin): return buf
    buf += "\n## Stability & Resonant Basins\n"
    if has_heat:
        buf += "- Stability heatmap: `plots/stability_heatmap.png`\n"
    if has_basin:
        buf += "- WWI contours + stability hatch: `plots/resonant_basins.png`\n"
    buf += ("**Reading:** color = WWI (% vs ΛCDM on fσ8 with 1-param amplitude fit); "
            "white contours at 50/70/85; hatched regions pass stability (bounded D, low TV_D, low stress).\n")
    return buf

def main():
    # 1) (Re)score ISW amplitudes and make figure if plotter exists
    maybe_run_py("tools/score_isw_from_csv.py", [])
    maybe_run_py("plots/plot_isw_amplitude.py", [])

    # 2) Update REPORT.md with fs8 + ISW + stability/basins
    buf = ensure_report()
    buf = append_fs8_section(buf)
    buf = append_isw_amp_section(buf)
    buf = append_stability_sections(buf)
    REPORT.write_text(buf)
    print(f"Updated {REPORT}")

    # 3) Print a compact console summary
    Fs = read_json("outputs/phase2/fs8_eval.json") or {}
    Isw = read_json("outputs/phase2/isw_score.json") or {}
    print("\n=== Phase-2 Finalize Summary ===")
    if Fs:
        print(f"fs8 χ²  {fmt(Fs.get('chi2'))}  (WWI={Fs.get('WWI')})")
    if Isw:
        print(f"ISW χ²  {fmt(Isw.get('chi2'))}  (WWI={Isw.get('WWI')})")
    print("Artifacts updated in outputs/phase2 and plots/")

if __name__ == "__main__":
    main()
