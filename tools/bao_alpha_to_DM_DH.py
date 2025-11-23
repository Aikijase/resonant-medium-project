#!/usr/bin/env python3
import argparse, csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--alphas", required=True, help="CSV with z,alpha_perp,alpha_par,sigma_perp,sigma_par")
ap.add_argument("--fiducial", required=True, help="CSV with z,DM_over_rd_fid,DH_over_rd_fid")
ap.add_argument("--out", default="data/bao_from_alpha.csv")
args = ap.parse_args()

# read fid table
fid = {}
with open(args.fiducial, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        z = float(row["z"])
        fid[z] = (float(row["DM_over_rd_fid"]), float(row["DH_over_rd_fid"]))

rows = []
with open(args.alphas, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        z = float(row["z"])
        ap_ = float(row["alpha_perp"])
        al_ = float(row["alpha_par"])
        sp  = float(row.get("sigma_perp", "0.0") or 0.0)
        sl  = float(row.get("sigma_par", "0.0") or 0.0)
        if z not in fid: continue
        dm_fid, dh_fid = fid[z]
        dm = ap_ * dm_fid
        dh = al_ * dh_fid
        rows.append({"z":z, "DM_over_rd":dm, "err_DM_over_rd":sp*dm_fid,
                          "DH_over_rd":dh, "err_DH_over_rd":sl*dh_fid})

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["z","DM_over_rd","err_DM_over_rd","DH_over_rd","err_DH_over_rd"])
    w.writeheader(); w.writerows(rows)
print(f"Wrote {args.out}")
