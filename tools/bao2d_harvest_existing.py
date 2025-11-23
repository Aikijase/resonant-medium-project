# tools/bao2d_harvest_existing.py
import json, csv, pathlib, glob, os

OUTDIR = pathlib.Path("outputs/bao2d_json"); OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / "summary.csv"

def try_float(x):
    try: return float(x)
    except: return None

rows = []
# Scan any plausible result JSONs you already have
CAND_PATTERNS = [
    "outputs/joint_*.json",
    "outputs/*/joint_*.json",
    "outputs/*.json",
]
seen = set()
for pat in CAND_PATTERNS:
    for p in glob.glob(pat):
        if p in seen: continue
        seen.add(p)
        try:
            with open(p) as fh:
                J = json.load(fh)
        except Exception:
            continue

        # schema tolerant
        bp = J.get("best_params") or J.get("params") or {}
        chi2 = J.get("chi2") or J.get("chi_sq") or J.get("chisq") or J.get("chi2_min")
        chi2 = try_float(chi2)

        g = try_float(bp.get("gamma", J.get("gamma")))
        f = try_float(bp.get("f", J.get("f")))
        phi = try_float(bp.get("phi", J.get("phi")))
        A = try_float(bp.get("A", J.get("A")))

        # Need at least gamma and chi2 to be useful
        if chi2 is None or g is None:
            continue

        rows.append({
            "gamma": g, "f": f, "chi2": chi2, "phi": phi, "A": A, "json": os.path.relpath(p)
        })

# Sort for readability
rows.sort(key=lambda r: (r["gamma"], (r["f"] if r["f"] is not None else 1e9), r["chi2"]))

if not rows:
    raise SystemExit("No usable JSON files found under outputs/. Did the runner write results?")

with open(OUTCSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["gamma","f","chi2","phi","A","json"])
    w.writeheader(); w.writerows(rows)

print("Wrote", OUTCSV, "with", len(rows), "rows.")
