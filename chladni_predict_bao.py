#!/usr/bin/env python3
import argparse, numpy as np, csv, json, math, sys
def omega_of_z(z, law, O0, O1):
    if law=="const": return O0
    if law=="z_over_1pz": return O0 + O1*(z/(1+z))
    if law=="ln1pz": return O0 + O1*math.log(1+z)
    if law=="power": return O0 + O1*((1+z)**0.5 - 1.0)
    raise SystemExit(f"unknown law={law}")
ap=argparse.ArgumentParser()
ap.add_argument('--a', type=float, default=0.078, help='slope from calibration: k ≈ a Ω + b')
ap.add_argument('--b', type=float, default=-0.011, help='intercept from calibration')
ap.add_argument('--O0', type=float, default=1.6, help='Ω at z=0')
ap.add_argument('--O1', type=float, default=0.8, help='variation amplitude')
ap.add_argument('--law', choices=['const','z_over_1pz','ln1pz','power'], default='z_over_1pz')
ap.add_argument('--zmin', type=float, default=0.1)
ap.add_argument('--zmax', type=float, default=2.0)
ap.add_argument('--dz', type=float, default=0.05)
ap.add_argument('--zref', type=float, default=0.70)
ap.add_argument('--out', type=str, default='outputs/chladni_bao_pred.csv')
args=ap.parse_args()

zs=np.arange(args.zmin, args.zmax+1e-9, args.dz)
def k_of_z(z): return args.a*omega_of_z(z, args.law, args.O0, args.O1)+args.b
def spacing(z): 
    k=k_of_z(z)
    return float('nan') if k<=0 else 1.0/k

sref=spacing(args.zref)
with open(args.out, 'w', newline='') as f:
    w=csv.writer(f)
    w.writerow(['z','Omega','k_pred','spacing_pred','relative_spacing'])
    for z in zs:
        Om=omega_of_z(z, args.law, args.O0, args.O1)
        k=k_of_z(z); s=spacing(z)
        rel=(s/sref) if (sref==sref and s==s) else float('nan')
        w.writerow([f"{z:.4f}", f"{Om:.6f}", f"{k:.6f}", f"{s:.6f}", f"{rel:.6f}"])
print("Saved", args.out)
