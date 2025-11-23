#!/usr/bin/env python3
import argparse, csv, os
import math

def main():
    ap=argparse.ArgumentParser(description="Convert band LF to bolometric using provided k-corr factors per row")
    ap.add_argument("agn_band_lf_csv", help="CSV with z, log10_Lband_erg_s, phi_dex_Mpc3, ..., band, k_corr_ref, ...")
    ap.add_argument("--kbol-col", default=None, help="Optional column index (0-based) with kbol factor; if None uses 10 for X-ray, 3.8 for optical (rough defaults)")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    with open(args.agn_band_lf_csv, newline="") as f:
        rows=[r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, data = rows[0], rows[1:]
    out=[hdr]  # keep same header names
    for r in data:
        band = r[4] if len(r)>4 else "bolometric"
        try:
            logL=float(r[1])
        except:
            out.append(r); continue
        if band.lower() == "bolometric":
            out.append(r); continue
        # Simple default kbol factors (placeholder; replace with Lusso+12 or Hopkins+07 tables as needed)
        kbol = 10.0 if band.lower()=="xray" else 3.8 if band.lower()=="optical" else 8.0
        if args.kbol_col is not None:
            try:
                kbol = float(r[int(args.kbol_col)])
            except: pass
        logLbol = math.log10((10**logL)*kbol)
        r2 = list(r); r2[1] = f"{logLbol:.6f}"; r2[4] = "bolometric"
        out.append(r2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",newline="") as f:
        csv.writer(f).writerows(out)
    print("Wrote", args.out, "rows=", len(out)-1)

if __name__=="__main__":
    main()
