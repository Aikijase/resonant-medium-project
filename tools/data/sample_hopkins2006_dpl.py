#!/usr/bin/env python3

"""
Sample Hopkins et al. (2006) bolometric QLF using a double power-law (DPL).
Form (per Hopkins+06, Eq. 6 in your PDF):
   phi(L) = dPhi/dlog10L = phi_star / [ (L/L_star)^gamma1 + (L/L_star)^gamma2 ]

INPUT PARAM CSV (you provide):
  z, log10Lstar_erg_s, log10phistar_Mpc3dex, gamma1, gamma2

We intentionally make z-dependence explicit via rows in the param file.
This avoids hard-coding any specific redshift evolution model and stays faithful to the paper.

Usage:
  python3 tools/data/sample_hopkins2006_dpl.py params/hopkins2006_params.csv --z 0.5 1.0 2.0 3.0 \
      --logL 44 47 15 --out data/hopkins2006_bolqlf_samples.csv
Meaning of --logL: start end Npoints  (log10 erg/s)

It will also append rows to templates/hopkins2006_bolqlf_digitize.csv for easy review.
"""
import argparse, csv, os, math

def dpl_phi(logL, logLstar, logphistar, g1, g2):
    L = 10.0**logL
    Lstar = 10.0**logLstar
    phistar = 10.0**logphistar
    return phistar / ( (L/Lstar)**g1 + (L/Lstar)**g2 )

def load_params(path):
    with open(path, newline="") as f:
        rows=[r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, data = rows[0], rows[1:]
    name_to_idx = {n:i for i,n in enumerate(hdr)}
    out = {}
    for r in data:
        z = float(r[name_to_idx["z"]])
        out[z] = {
            "logLstar": float(r[name_to_idx["log10Lstar_erg_s"]]),
            "logphistar": float(r[name_to_idx["log10phistar_Mpc3dex"]]),
            "g1": float(r[name_to_idx["gamma1"]]),
            "g2": float(r[name_to_idx["gamma2"]]),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("param_csv")
    ap.add_argument("--z", nargs="+", type=float, required=True)
    ap.add_argument("--logL", nargs=3, type=float, metavar=("START","END","N"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append-digitize", default="templates/hopkins2006_bolqlf_digitize.csv")
    args = ap.parse_args()

    params = load_params(args.param_csv)
    Ls = [args.logL[0] + i*(args.logL[1]-args.logL[0])/(args.logL[2]-1) for i in range(int(args.logL[2]))]
    rows_out = [["z","log10_Lbol_erg_s","phi_Mpc3_dex","source_id","note"]]

    for z in args.z:
        if z not in params:
            # nearest z in params
            z_near = min(params.keys(), key=lambda zz: abs(zz-z))
        else:
            z_near = z
        p = params[z_near]
        for logL in Ls:
            phi = dpl_phi(logL, p["logLstar"], p["logphistar"], p["g1"], p["g2"])
            rows_out.append([z, f"{logL:.3f}", f"{phi:.6e}", "HOPKINS06", f"params@z={z_near}"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows_out)

    # Append to digitize worksheet for quick review
    if args.append_digitize and os.path.exists(args.append_digitize):
        with open(args.append_digitize, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows_out[1:]:
                z, logL, phi, sid, note = r
                w.writerow([z, "", "", logL, phi, "", 1, "bolometric", "fit-sample", sid, note])

    print("Wrote", args.out, "rows=", len(rows_out)-1)

if __name__ == "__main__":
    main()
