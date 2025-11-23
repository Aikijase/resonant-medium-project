#!/usr/bin/env python3
"""
Sample Hopkins et al. (2006) bolometric QLF using a double power-law (DPL):
  dPhi/dlog10L = phi* / ( (L/L*)^gamma1 + (L/L*)^gamma2 )

INPUT param CSV you provide: z, log10Lstar_erg_s, log10phistar_Mpc3dex, gamma1, gamma2
Usage:
  python3 tools/data/sample_hopkins2006_dpl.py params/hopkins2006_params.csv \
    --z 0.5 1.0 2.0 3.0 --logL 44 47 15 \
    --out data/hopkins2006_bolqlf_samples.csv
Also appends rows to templates/hopkins2006_bolqlf_digitize.csv if it exists.
"""
import argparse, csv, os, math

def dpl_phi(logL, logLstar, logphistar, g1, g2):
    L, Lstar, phistar = 10.0**logL, 10.0**logLstar, 10.0**logphistar
    return phistar / ( (L/Lstar)**g1 + (L/Lstar)**g2 )

def load_params(path):
    with open(path, newline="") as f:
        rows=[r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, data = rows[0], rows[1:]
    idx = {n:i for i,n in enumerate(hdr)}
    out = {}
    for r in data:
        try:
            z=float(r[idx["z"]])
        except: 
            continue
        out[z] = {
            "logLstar": float(r[idx["log10Lstar_erg_s"]]),
            "logphistar": float(r[idx["log10phistar_Mpc3dex"]]),
            "g1": float(r[idx["gamma1"]]),
            "g2": float(r[idx["gamma2"]]),
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
    out = [["z","log10_Lbol_erg_s","phi_Mpc3_dex","source_id","note"]]
    for z in args.z:
        zkey = min(params.keys(), key=lambda zz: abs(zz-z))
        p = params[zkey]
        for logL in Ls:
            phi = dpl_phi(logL, p["logLstar"], p["logphistar"], p["g1"], p["g2"])
            out.append([z, f"{logL:.3f}", f"{phi:.6e}", "HOPKINS06", f"params@z={zkey}"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(out)

    if args.append_digitize and os.path.exists(args.append_digitize):
        with open(args.append_digitize, "a", newline="") as f:
            w = csv.writer(f)
            for r in out[1:]:
                z, logL, phi, sid, note = r
                w.writerow([z, "", "", logL, phi, "", 1, "bolometric", "fit-sample", sid, note])
    print("Wrote", args.out, "rows=", len(out)-1)

if __name__ == "__main__":
    main()
