#!/usr/bin/env python3
import argparse, csv, os

def load_csv(path):
    with open(path, newline='') as f:
        rows=[r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, data = rows[0], rows[1:]
    return hdr, data

def main():
    ap=argparse.ArgumentParser(description="Merge Aird+2015, Ueda+2014, Hopkins+2006 LF points into a single table")
    ap.add_argument("--aird"); ap.add_argument("--ueda"); ap.add_argument("--hopkins")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    out_hdr = ["source","z_center","z_low","z_high","log10_L","phi_Mpc3_dex","sigma_phi","band_or_bol","absorbed_flag","fig_panel","notes"]
    out_rows = [out_hdr]

    def append_rows(label, hdr, data, is_bol=False):
        # Map columns by name heuristically
        name_to_idx = {n:i for i,n in enumerate(hdr)}
        for r in data:
            try:
                if is_bol:
                    L = r[name_to_idx.get("log10_Lbol_erg_s", 3)]
                    band = "bolometric"
                else:
                    L = r[name_to_idx.get("log10_Lx_erg_s", 3)]
                    band = r[name_to_idx.get("band", 6)]
                zc = r[name_to_idx.get("z_bin_center", 0)]
                zl = r[name_to_idx.get("z_bin_low", 1)]
                zh = r[name_to_idx.get("z_bin_high", 2)]
                phi = r[name_to_idx.get("phi_Mpc3_dex", 4)]
                sig = r[name_to_idx.get("sigma_phi", 5)]
                absf = r[name_to_idx.get("absorbed_flag", 7)] if not is_bol else ""
                figp = r[name_to_idx.get("fig_panel", 10 if not is_bol else 8)]
                notes = r[name_to_idx.get("notes", 12 if not is_bol else 10)]
                out_rows.append([label, zc, zl, zh, L, phi, sig, band, absf, figp, notes])
            except Exception as e:
                # Skip malformed row
                continue

    if args.aird and os.path.exists(args.aird):
        hdr, data = load_csv(args.aird); append_rows("Aird2015", hdr, data, is_bol=False)
    if args.ueda and os.path.exists(args.ueda):
        hdr, data = load_csv(args.ueda); append_rows("Ueda2014", hdr, data, is_bol=False)
    if args.hopkins and os.path.exists(args.hopkins):
        hdr, data = load_csv(args.hopkins); append_rows("Hopkins2006", hdr, data, is_bol=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    print("Wrote", args.out, "rows=", len(out_rows)-1)

if __name__=="__main__":
    main()
