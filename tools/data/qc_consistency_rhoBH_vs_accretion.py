#!/usr/bin/env python3
import argparse, csv, math, os

def read_series(path):
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        rows=[r for r in rdr if r and not r[0].startswith("#")]
    hdr, data = rows[0], rows[1:]
    zs=[]; vals=[]
    for r in data:
        try:
            zs.append(float(r[0])); vals.append(float(r[1]))
        except: pass
    return zs, vals

def trapezoid(x, y):
    s=0.0
    for i in range(len(x)-1):
        dx = x[i+1]-x[i]
        s += 0.5*(y[i]+y[i+1])*dx
    return s

def main():
    ap=argparse.ArgumentParser(description="Consistency check: Δrho_BH vs ∫ rho_dot_acc dt")
    ap.add_argument("--rho-bh", required=True, help="CSV with z, rho_bh_Msun_Mpc3")
    ap.add_argument("--rho-dot", required=True, help="CSV with z, rho_dot_acc_Msun_yr_Mpc3")
    ap.add_argument("--H0", type=float, default=67.4, help="H0 for dt/dz conversion (km/s/Mpc)")
    ap.add_argument("--Omega_m", type=float, default=0.315)
    ap.add_argument("--Omega_L", type=float, default=0.685)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    # Cosmology helper (flat)
    H0 = args.H0 * 1e5 / 3.085677581e24  # s^-1
    Om, Ol = args.Omega_m, args.Omega_L

    def E(z): return math.sqrt(Om*(1+z)**3 + Ol)
    def dt_dz(z): return -1.0 / ((1+z)*H0*E(z))  # s per unit z

    zb, rb = read_series(args.rho_bh)
    za, rdot = read_series(args.rho_dot)

    if not zb or not rb:
        raise SystemExit(f"ERROR: {args.rho_bh} has no numeric rows; fill/ingest ρ_BH first.")
    if not za or not rdot:
        raise SystemExit(f"ERROR: {args.rho_dot} has no numeric rows; check LF integration input.")
    # Ensure ascending z
    rb = [x for _,x in sorted(zip(zb,rb))]
    zb = sorted(zb)
    rdot = [x for _,x in sorted(zip(za,rdot))]
    za = sorted(za)

    # Interpolate rdot onto zb
    import bisect
    def interp(x, xs, ys):
        if x<=xs[0]: return ys[0]
        if x>=xs[-1]: return ys[-1]
        i=bisect.bisect_left(xs, x)
        x0,x1=xs[i-1],xs[i]; y0,y1=ys[i-1],ys[i]
        t=(x-x0)/(x1-x0); return y0 + t*(y1-y0)

    rdot_on_b = [interp(z, za, rdot) for z in zb]
    # Convert rdot (Msun/yr/Mpc^3) to Msun/Mpc^3 per unit z via dt/dz
    # year in seconds:
    sec_per_year = 365.25*24*3600.0
    # Convert Msun/yr/Mpc^3 to Msun/Mpc^3 per unit z: multiply by dt/dz (s/z) and divide by sec/year
    drho_dz = [rd*(dt_dz(z)/sec_per_year) for z, rd in zip(zb, rdot_on_b)]
    delta_rho = - trapezoid(zb, drho_dz)  # flip sign so Δρ>0 when integrating from high→low z
    # Compare with observed rho_BH span
    observed_delta = rb[0] - rb[-1]  # assuming zb ascending (low z first)
    ratio = (delta_rho/observed_delta) if observed_delta!=0 else float('nan')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["delta_rho_from_accretion_Msun_Mpc3", f"{delta_rho:.6e}"])
        w.writerow(["observed_delta_rhoBH_Msun_Mpc3", f"{observed_delta:.6e}"])
        w.writerow(["ratio_accretion_to_observed", f"{ratio:.6f}"])
    print("Wrote", args.out)

if __name__=="__main__":
    main()
