#!/usr/bin/env python3
# tools/data/clean_rho_bh_series.py
# Usage: python3 clean_rho_bh_series.py INPUT.csv OUTPUT.csv
import csv, sys, math

if len(sys.argv) != 3:
    print("Usage: python3 clean_rho_bh_series.py INPUT.csv OUTPUT.csv", file=sys.stderr)
    sys.exit(2)

inp, outp = sys.argv[1], sys.argv[2]

rows = []
with open(inp, newline="") as f:
    sn = csv.Sniffer()
    sample = f.read(2048)
    f.seek(0)
    has_header = sn.has_header(sample)
    rdr = csv.reader(f)
    header = next(rdr) if has_header else None

    # Try to find z and rho columns by name first
    z_idx = rho_idx = None
    if header:
        for i, name in enumerate(header):
            n = (name or "").lower()
            if z_idx is None and ("z" == n or n.startswith("z_")):
                z_idx = i
            if rho_idx is None and ("rho" in n and ("bh" in n or "msun" in n or "mpc" in n)):
                rho_idx = i

    # Fallback: assume first two columns are z, rho if not detected
    if z_idx is None: z_idx = 0
    if rho_idx is None: rho_idx = 1

    for r in rdr:
        if not r or len(r) <= max(z_idx, rho_idx): 
            continue
        try:
            z = float(r[z_idx])
            rho = float(r[rho_idx])
            if math.isfinite(z) and math.isfinite(rho):
                rows.append((z, rho))
        except Exception:
            continue

# De-dup by z, keep last
d = {}
for z, rho in rows:
    d[z] = rho
rows = sorted(d.items(), key=lambda t: t[0])

if not rows:
    print("ERROR: No numeric rows found after cleaning.", file=sys.stderr)
    sys.exit(1)

with open(outp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["z", "rho_bh_Msun_per_Mpc3"])
    w.writerows(rows)

print(f"Wrote {outp} rows={len(rows)} from {inp}")
