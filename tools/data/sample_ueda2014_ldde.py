#!/usr/bin/env python3

"""
Sample Ueda et al. (2014) XLF using an LDDE-style form.
This is a scaffold: you paste the Table 4 parameters into params/ueda2014_ldde_params.csv and
the script will evaluate phi(Lx, z) on a grid.

Generic LDDE-like shape (see Ueda+14 for exact formulation):
  phi(L,z) = A / [ (L/L*)^γ1 + (L/L*)^γ2 ] * e(z, L)
  with e(z,L) = (1+z)^{p1} for z <= zc(L), and (1+zc)^{p1} * ((1+z)/(1+zc))^{p2} above zc
  and zc(L) = zc* * (L / L_a)^α for L < L_a and zc(L)=zc* for L >= L_a

INPUT PARAM CSV (example header; fill from Table 4 exactly!):
  zlist, log10Lstar_erg_s, log10A_Mpc3dex, gamma1, gamma2, p1, p2, zc_star, log10La_erg_s, alpha

Usage:
  python3 tools/data/sample_ueda2014_ldde.py params/ueda2014_ldde_params.csv --z 0.1 0.5 1 2 3 \
      --logL 42 46 15 --out data/ueda2014_xlf_samples.csv

This will also append rows to templates/ueda2014_xlf_digitize.csv for easy review.
"""
import argparse, csv, os, math

def ldde_phi(logL, z, p):
    L = 10.0**logL
    Lstar = 10.0**p["logLstar"]
    A = 10.0**p["logA"]
    g1, g2 = p["g1"], p["g2"]
    La = 10.0**p["logLa"]
    zc_star = p["zc_star"]
    alpha = p["alpha"]
    p1, p2 = p["p1"], p["p2"]

    # DPL backbone
    phi0 = A / ( (L/Lstar)**g1 + (L/Lstar)**g2 )

    # zc(L)
    zc = zc_star if L >= La else zc_star * (L/La)**alpha

    # evolution
    if z <= zc:
        evol = (1.0+z)**p1
    else:
        evol = (1.0+zc)**p1 * ((1.0+z)/(1.0+zc))**p2

    return phi0 * evol

def load_params(path):
    with open(path, newline="") as f:
        rows=[r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, data = rows[0], rows[1:]
    idx = {n:i for i,n in enumerate(hdr)}
    # We allow a single set of params to be used across all z in --z
    # or multiple rows keyed by e.g. a named zlist (comma-separated)
    p = {
        "logLstar": float(data[0][idx["log10Lstar_erg_s"]]),
        "logA": float(data[0][idx["log10A_Mpc3dex"]]),
        "g1": float(data[0][idx["gamma1"]]),
        "g2": float(data[0][idx["gamma2"]]),
        "p1": float(data[0][idx["p1"]]),
        "p2": float(data[0][idx["p2"]]),
        "zc_star": float(data[0][idx["zc_star"]]),
        "logLa": float(data[0][idx["log10La_erg_s"]]),
        "alpha": float(data[0][idx["alpha"]]),
    }
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("param_csv")
    ap.add_argument("--z", nargs="+", type=float, required=True)
    ap.add_argument("--logL", nargs=3, type=float, metavar=("START","END","N"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append-digitize", default="templates/ueda2014_xlf_digitize.csv")
    args = ap.parse_args()

    params = load_params(args.param_csv)
    Ls = [args.logL[0] + i*(args.logL[1]-args.logL[0])/(args.logL[2]-1) for i in range(int(args.logL[2]))]

    rows_out = [["z","log10_Lx_erg_s","phi_Mpc3_dex","source_id","note"]]
    for z in args.z:
        for logL in Ls:
            phi = ldde_phi(logL, z, params)
            rows_out.append([z, f"{logL:.3f}", f"{phi:.6e}", "UEDA14", "LDDE scaffold; verify with Table 4"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows_out)

    # Append to digitize worksheet
    if args.append_digitize and os.path.exists(args.append_digitize):
        with open(args.append_digitize, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows_out[1:]:
                z, logL, phi, sid, note = r
                w.writerow([z, "", "", logL, phi, "", "Xray", "", "fit-sample", "LDDE", "UEDA14", note])

    print("Wrote", args.out, "rows=", len(rows_out)-1)

if __name__ == "__main__":
    main()
