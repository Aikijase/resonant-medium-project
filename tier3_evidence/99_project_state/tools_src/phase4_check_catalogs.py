#!/usr/bin/env python3
"""
Validate data/isw/gk_catalogs.csv for the g×κ scorer.

Checks:
- required columns exist: name,z_eff,A_lit,sigma_lit
- numeric fields parse and are finite
- sigma_lit > 0
- duplicate (name,z_eff) pairs
- optional columns allowed: notes

Prints a brief report and exits non-zero if problems are found.
"""
import csv, math, sys

CSV_PATH = "data/isw/gk_catalogs.csv"
REQUIRED = ["name","z_eff","A_lit","sigma_lit"]

def is_finite(x):
    try:
        x = float(x)
        return math.isfinite(x)
    except Exception:
        return False

def main():
    try:
        f = open(CSV_PATH, newline="")
    except Exception as e:
        print(f"[error] cannot open {CSV_PATH}: {e}", file=sys.stderr)
        sys.exit(2)

    reader = csv.DictReader(f)
    missing = [c for c in REQUIRED if c not in reader.fieldnames]
    if missing:
        print(f"[error] missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    errors = 0
    warnings = 0
    seen = set()
    nrows = 0

    for i, row in enumerate(reader, 2):  # header is line 1
        nrows += 1
        name = (row.get("name") or "").strip()
        z_eff = row.get("z_eff")
        A_lit = row.get("A_lit")
        s_lit = row.get("sigma_lit")

        # presence
        if not name:
            print(f"[error] line {i}: empty 'name'", file=sys.stderr); errors += 1
        # numeric checks
        if not is_finite(z_eff):
            print(f"[error] line {i}: z_eff not finite → {z_eff}", file=sys.stderr); errors += 1
        if not is_finite(A_lit):
            print(f"[error] line {i}: A_lit not finite → {A_lit}", file=sys.stderr); errors += 1
        if not is_finite(s_lit):
            print(f"[error] line {i}: sigma_lit not finite → {s_lit}", file=sys.stderr); errors += 1
        else:
            s = float(s_lit)
            if s <= 0:
                print(f"[error] line {i}: sigma_lit must be > 0 (got {s})", file=sys.stderr); errors += 1
            if s > 0.5:
                print(f"[warn]  line {i}: large sigma_lit={s} (check units)", file=sys.stderr); warnings += 1

        # duplicates
        try:
            key = (name, float(z_eff))
            if key in seen:
                print(f"[warn]  line {i}: duplicate (name,z_eff)={key}", file=sys.stderr); warnings += 1
            seen.add(key)
        except Exception:
            pass

    print(f"[ok] scanned {nrows} rows; warnings={warnings}, errors={errors}")
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
