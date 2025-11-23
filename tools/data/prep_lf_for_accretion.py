#!/usr/bin/env python3
import csv, argparse, os

def is_float(x):
    try: float(x); return True
    except: return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-bolometric", action="store_true")
    args = ap.parse_args()

    with open(args.in_csv, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    hdr, data = rows[0], rows[1:]
    name = {n.strip(): i for i, n in enumerate(hdr)}

    z_col   = name.get("z_center", name.get("z"))
    L_col   = name.get("log10_L", name.get("log10_Lbol_erg_s"))
    phi_col = name.get("phi_Mpc3_dex", name.get("phi_dex_Mpc3"))
    band_col = name.get("band_or_bol", name.get("band"))

    if None in (z_col, L_col, phi_col):
        raise SystemExit(f"Needed columns not found in header: {hdr}")

    out = [["z","log10_Lbol_erg_s","phi_dex_Mpc3"]]
    kept = 0
    for r in data:
        if args.require_bolometric and band_col is not None:
            b = (r[band_col] or "").strip().lower()
            if b != "bolometric":
                continue
        z, L, phi = r[z_col], r[L_col], r[phi_col]
        if not (is_float(z) and is_float(L) and is_float(phi)):
            continue
        out.append([f"{float(z):.6f}", f"{float(L):.6f}", f"{float(phi):.6e}"])
        kept += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"Wrote {args.out} rows= {kept}")
if __name__ == "__main__":
    main()
