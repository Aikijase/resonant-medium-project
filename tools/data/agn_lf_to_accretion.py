#!/usr/bin/env python3
import argparse, csv, os, math

def read_rows(path):
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        rows=[r for r in rdr if r and not r[0].startswith("#")]
    hdr, data = rows[0], rows[1:]
    return hdr, data

def group_by_z(data, zcol=0):
    byz={}
    for r in data:
        try:
            z=float(r[zcol])
        except: 
            continue
        byz.setdefault(z, []).append(r)
    return dict(sorted(byz.items()))

def main():
    ap=argparse.ArgumentParser(description="Integrate AGN bolometric LF to BH accretion rate density via Soltan argument")
    ap.add_argument("agn_bol_lf_csv", help="CSV with columns: z, log10_Lbol_erg_s, phi_dex_Mpc3, ...")
    ap.add_argument("--epsilon", type=float, default=0.1, help="Radiative efficiency (dimensionless)")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    c=2.99792458e10  # cm/s
    erg_per_Msun = 1.98847e33 * c*c  # erg per Msun*c^2
    hdr, data = read_rows(args.agn_bol_lf_csv)
    # Expected: z, log10_Lbol_erg_s, phi_dex_Mpc3
    byz = group_by_z(data, zcol=0)
    out=[["z","rho_dot_acc_Msun_yr_Mpc3","sigma_rho_dot","method","band","bol_corr_ref","source_id","notes"]]
    for z, rows in byz.items():
        # sort by L and integrate \int L * phi(L) d log10 L * ln(10)
        pairs=[]
        for r in rows:
            try:
                L=10.0**float(r[1])
                phi=float(r[2])
            except:
                continue
            pairs.append((L,phi))
        pairs.sort()
        rhoL = 0.0
        for i in range(len(pairs)-1):
            L0,phi0 = pairs[i]
            L1,phi1 = pairs[i+1]
            # trapezoid in log10L: integrate L*phi d(log10L)
            logL0=math.log10(L0); logL1=math.log10(L1)
            dlogL = (logL1-logL0)
            # average L*phi over bin edges
            rhoL += 0.5*((L0*phi0) + (L1*phi1))*dlogL*math.log(10.0)
        # Convert luminosity density to mass accretion rate density via rhoL = epsilon * rho_dot * c^2
        rho_dot = rhoL / (args.epsilon * erg_per_Msun)  # Msun/s/Mpc^3
        rho_dot *= (3600*24*365.25)  # -> Msun/yr/Mpc^3
        out.append([z, f"{rho_dot:.6e}", "", "LF_integration", "bolometric", "", "", ""])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",newline="") as f:
        csv.writer(f).writerows(out)
    print("Wrote", args.out, "rows=", len(out)-1)

if __name__=="__main__":
    main()
