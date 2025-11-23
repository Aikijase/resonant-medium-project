#!/usr/bin/env python3
"""
Batch runner for phase14_contour_width.py:
- Executes multiple (low, high) Si pairs
- Parses console metrics robustly (floats or ints in stdout)
- Writes a summary CSV + JSON

Example:
  python3 tools/phase14_batch_widths.py \
    --shell-csv outputs/phase14/p14_edge_shell02_results.csv \
    --outdir outputs/phase14 \
    --prefix p14_edge \
    --levels 0.90,0.92,0.94,0.96,0.98 \
    --pairs 0.90:0.98,0.92:0.98,0.94:0.98
"""

import argparse, json, os, re, subprocess, sys
from datetime import datetime

PAIR_RE = re.compile(r"^\s*([0-9.]+)\s*:\s*([0-9.]+)\s*$")

def safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

def safe_intlike(s, default=None):
    try:
        return int(float(s))
    except Exception:
        return default

def run_pair(tool, shell_csv, outdir, prefix, levels, low, high, nw, nk, wbins):
    cmd = [
        sys.executable, tool,
        "--shell-csv", shell_csv,
        "--outdir", outdir,
        "--prefix", prefix,
        "--levels", levels,
        "--low", str(low),
        "--high", str(high),
        "--nw", str(nw),
        "--nk", str(nk),
        "--wbins", str(wbins),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"low": low, "high": high, "status": "error", "stderr": p.stderr.strip()}

    stdout = p.stdout

    def grab(pattern, default=None):
        m = re.search(pattern, stdout, flags=re.MULTILINE)
        return m.group(1).strip() if m else default

    def grab_path(label):
        m = re.search(rf"{label}:\s*(.+)$", stdout, flags=re.MULTILINE)
        return m.group(1).strip() if m else None

    # Handle both ASCII and Unicode arrows for omega2 span
    omega2_min_s = grab(r"omega2_span_used \(trapz\):\s*([0-9.eE+-]+)")
    omega2_max_s = grab(r"omega2_span_used \(trapz\):\s*[0-9.eE+-]+\s*(?:->|→)\s*([0-9.eE+-]+)")

    res = {
        "low": low,
        "high": high,
        "status": "ok",
        "contour_png": grab_path("contour_png"),
        "width_png": grab_path("width_png"),
        "mask_png": grab_path("mask_png"),
        "width_csv": grab_path("width_envelope_csv"),
        "merge_mode": (grab(r"merge_mode:\s*(.+)$") or "strict-bin"),
        "mean_width": safe_float(grab(r"mean'\:\s*([0-9.eE+-]+)")),
        "count": safe_intlike(grab(r"count'\:\s*([0-9.eE+-]+)")),
        "std_width": safe_float(grab(r"std'\:\s*([0-9.eE+-]+)")),
        "min_width": safe_float(grab(r"min'\:\s*([0-9.eE+-]+)")),
        "q25_width": safe_float(grab(r"25%'\:\s*([0-9.eE+-]+)")),
        "median_width": safe_float(grab(r"50%'\:\s*([0-9.eE+-]+)")),
        "q75_width": safe_float(grab(r"75%'\:\s*([0-9.eE+-]+)")),
        "max_width": safe_float(grab(r"max'\:\s*([0-9.eE+-]+)")),
        "A_trapz": safe_float(grab(r"A_trapz \(envelope\)\s*:\s*([0-9.eE+-]+)")),
        "A_mask": safe_float(grab(r"A_mask\s*\(grid band\)\s*:\s*([0-9.eE+-]+)")),
        "omega2_min": safe_float(omega2_min_s),
        "omega2_max": safe_float(omega2_max_s),
        "stdout_raw": stdout.strip(),
    }
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="tools/phase14_contour_width.py")
    ap.add_argument("--shell-csv", required=True)
    ap.add_argument("--outdir", default="outputs/phase14")
    ap.add_argument("--prefix", default="p14_edge")
    ap.add_argument("--levels", default="0.90,0.92,0.94,0.96,0.98")
    ap.add_argument("--pairs", default="0.90:0.98,0.92:0.98,0.94:0.98")
    ap.add_argument("--nw", type=int, default=400)
    ap.add_argument("--nk", type=int, default=400)
    ap.add_argument("--wbins", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pairs = []
    for s in args.pairs.split(","):
        m = PAIR_RE.match(s)
        if not m:
            print(f"Bad pair spec: {s}", file=sys.stderr); sys.exit(2)
        pairs.append((float(m.group(1)), float(m.group(2))))

    rows = []
    for low, high in pairs:
        rows.append(run_pair(args.tool, args.shell_csv, args.outdir, args.prefix,
                             args.levels, low, high, args.nw, args.nk, args.wbins))

    # Write summary CSV + JSON
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.outdir, f"{args.prefix}_batch_summary_{ts}.csv")
    json_path = os.path.join(args.outdir, f"{args.prefix}_batch_summary_{ts}.json")

    import csv
    keys = ["low","high","status","merge_mode","mean_width","A_trapz","A_mask",
            "omega2_min","omega2_max","count","std_width","min_width","median_width",
            "q25_width","q75_width","max_width","width_csv","width_png","mask_png"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("wrote summary CSV:", csv_path)
    print("wrote summary JSON:", json_path)
    print("pairs:", pairs)

if __name__ == "__main__":
    sys.exit(main())
