#!/usr/bin/env python3
"""
Phase 16 — Multi-shell comparison:
For each shell CSV:
  1) Run Phase-14 width/area (0.90→0.98 by default; can change via args)
  2) Build centerline & curvature (needs the two contour CSVs already written by step 1)
  3) Run Phase-15 ridge (from grid) & ridge–centerline delta stats
  4) Compute normalized area using the shell's domain
Outputs:
  - Per-shell JSON rows + a summary CSV/JSON
  - Optional combined comparison Markdown & LaTeX table
Assumes Phase-14/15 tools are already in tools/ as per your setup.
"""

import argparse, os, sys, json, subprocess, re, csv
import pandas as pd

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def grab_num(pattern, text, default=None):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else default

def grab_path(label, text):
    m = re.search(rf"{label}:\s*(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None

def domain_area(shell_csv):
    df = pd.read_csv(shell_csv).dropna(subset=["omega2","Kphi"])
    return float((df["omega2"].max()-df["omega2"].min())*(df["Kphi"].max()-df["Kphi"].min()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shells", required=True,
                    help="Comma-separated list of shell CSVs")
    ap.add_argument("--outdir", default="outputs/phase16")
    ap.add_argument("--prefix", default="p16")
    ap.add_argument("--levels", default="0.90,0.92,0.94,0.96,0.98")
    ap.add_argument("--low", type=float, default=0.90)
    ap.add_argument("--high", type=float, default=0.98)
    ap.add_argument("--nw", type=int, default=600)
    ap.add_argument("--nk", type=int, default=600)
    ap.add_argument("--wbins", type=int, default=900)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    shells = [s.strip() for s in args.shells.split(",") if s.strip()]

    rows = []
    for shell in shells:
        tag = os.path.splitext(os.path.basename(shell))[0]  # e.g., p14_edge_shell03_results
        p14_prefix = f"{tag}"
        out14 = os.path.join(os.path.dirname(shell))  # keep phase14 outputs next to shell
        # 1) Phase-14 widths & areas
        code, out, err = run([
            sys.executable, "tools/phase14_contour_width.py",
            "--shell-csv", shell,
            "--outdir", out14,
            "--prefix", p14_prefix,
            "--levels", args.levels,
            "--low", str(args.low), "--high", str(args.high),
            "--nw", str(args.nw), "--nk", str(args.nk), "--wbins", str(args.wbins),
        ])
        if code != 0:
            rows.append({"shell": shell, "status": "p14_error", "stderr": err})
            continue

        width_csv = grab_path("width_envelope_csv", out)
        width_png = grab_path("width_png", out)
        mask_png  = grab_path("mask_png", out)
        A_trapz   = grab_num(r"A_trapz \(envelope\)\s*:\s*([0-9.eE+-]+)", out)
        A_mask    = grab_num(r"A_mask\s*\(grid band\)\s*:\s*([0-9.eE+-]+)", out)
        w_mean    = grab_num(r"mean'\:\s*([0-9.eE+-]+)", out)
        omega2_min= grab_num(r"omega2_span_used \(trapz\):\s*([0-9.eE+-]+)", out)
        omega2_max= grab_num(r"omega2_span_used \(trapz\):\s*[0-9.eE+-]+\s*(?:->|→)\s*([0-9.eE+-]+)", out)
        contour_low  = os.path.join(out14, f"{p14_prefix}_contour_si{args.low:0.3f}.csv")
        contour_high = os.path.join(out14, f"{p14_prefix}_contour_si{args.high:0.3f}.csv")

        # 2) Phase-14 centerline
        code2, out2, err2 = run([
            sys.executable, "tools/phase14_centerline.py",
            "--contour-low",  contour_low,
            "--contour-high", contour_high,
            "--low", str(args.low), "--high", str(args.high),
            "--wbins", str(args.wbins),
            "--outdir", out14, "--prefix", p14_prefix
        ])
        if code2 != 0:
            rows.append({"shell": shell, "status": "centerline_error", "stderr": err2})
            continue
        center_csv = grab_path("centerline_csv", out2)
        center_png = grab_path("centerline_png", out2)
        c_center   = grab_num(r"curvature_c \(quadratic coefficient\):\s*([0-9.eE+-]+)", out2)

        # 3) Phase-15 ridge (+ delta vs centerline)
        out15_dir = os.path.join(args.outdir, tag)
        os.makedirs(out15_dir, exist_ok=True)
        code3, out3, err3 = run([
            sys.executable, "tools/phase15_ridge_from_grid.py",
            "--shell-csv", shell,
            "--centerline-csv", center_csv,
            "--outdir", out15_dir,
            "--prefix", "p15cmp",
            "--nw", str(args.nw), "--nk", str(args.nk),
        ])
        if code3 != 0:
            rows.append({"shell": shell, "status": "ridge_error", "stderr": err3})
            continue

        ridge_csv = grab_path("ridge_csv", out3)
        ridge_png = grab_path("ridge_png", out3)
        cmp_png   = grab_path("ridge_vs_centerline_png", out3)
        a = grab_num(r"ridge_quadratic_fit: a=([0-9\.\-eE]+)", out3); 
        b = grab_num(r"ridge_quadratic_fit: a=[^b]+b=([0-9\.\-eE]+)", out3)
        c = grab_num(r"ridge_quadratic_fit: .* c=([0-9\.\-eE]+)", out3)
        R2= grab_num(r"ridge_quadratic_fit: .* R2=([0-9\.\-eE]+)", out3)
        d_mean = grab_num(r"mean'\:\s*([0-9.eE+-]+)", out3)
        d_std  = grab_num(r"std'\:\s*([0-9.eE+-]+)", out3)
        d_min  = grab_num(r"min'\:\s*([0-9.eE+-]+)", out3)
        d_q25  = grab_num(r"25%'\:\s*([0-9.eE+-]+)", out3)
        d_med  = grab_num(r"50%'\:\s*([0-9.eE+-]+)", out3)
        d_q75  = grab_num(r"75%'\:\s*([0-9.eE+-]+)", out3)
        d_max  = grab_num(r"max'\:\s*([0-9.eE+-]+)", out3)

        # 4) Normalise
        A_dom = domain_area(shell)
        A_trapz_norm = (A_trapz / A_dom) if (A_trapz is not None and A_dom>0) else None
        A_mask_norm  = (A_mask  / A_dom) if (A_mask  is not None and A_dom>0) else None

        rows.append({
            "shell": shell, "status": "ok", "tag": tag,
            "mean_width": w_mean, "A_trapz": A_trapz, "A_mask": A_mask,
            "A_domain": A_dom, "A_trapz_norm": A_trapz_norm, "A_mask_norm": A_mask_norm,
            "omega2_min": omega2_min, "omega2_max": omega2_max,
            "centerline_csv": center_csv, "centerline_png": center_png, "c_center": c_center,
            "ridge_csv": ridge_csv, "ridge_png": ridge_png, "cmp_png": cmp_png,
            "ridge_a": a, "ridge_b": b, "ridge_c": c, "ridge_R2": R2,
            "d_mean": d_mean, "d_std": d_std, "d_min": d_min, "d_q25": d_q25,
            "d_med": d_med, "d_q75": d_q75, "d_max": d_max,
            "width_csv": width_csv, "width_png": width_png, "mask_png": mask_png
        })

    # Write summary
    js = os.path.join(args.outdir, f"{args.prefix}_summary.json")
    cs = os.path.join(args.outdir, f"{args.prefix}_summary.csv")
    with open(js,"w") as f: json.dump(rows, f, indent=2)

    keys = ["tag","status","mean_width","A_trapz","A_mask","A_domain","A_trapz_norm","A_mask_norm",
            "c_center","ridge_c","ridge_R2","d_mean","d_std","d_med","d_min","d_q25","d_q75","d_max"]
    with open(cs,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in keys})

    print("wrote:", cs)
    print("wrote:", js)
    print("shells:", shells)

if __name__ == "__main__":
    sys.exit(main())
