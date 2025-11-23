#!/usr/bin/env python3
import os, urllib.request

CANDIDATES = {
  "bao_measurements.csv": [
    "https://data.desi.lbl.gov/public/dr1/vac/bao/bao_DR1_measurements.csv",
    "https://data.desi.lbl.gov/public/dr1/vac/BAO/DR1/bao_DR1_measurements.csv",
    "https://data.desi.lbl.gov/public/dr1/vac/bao/bao_measurements.csv",
  ],
  "bao_covariance.csv": [
    "https://data.desi.lbl.gov/public/dr1/vac/bao/bao_DR1_covariance.csv",
    "https://data.desi.lbl.gov/public/dr1/vac/BAO/DR1/bao_DR1_covariance.csv",
    "https://data.desi.lbl.gov/public/dr1/vac/bao/bao_covariance.csv",
  ],
}

def ensure_dir(p): os.makedirs(os.path.dirname(p), exist_ok=True)

def try_fetch(urls, out):
    for u in urls:
        try:
            print(f"Trying {u}")
            urllib.request.urlretrieve(u, out)
            print(f"  -> saved to {out}")
            return True
        except Exception as e:
            print(f"  x {e}")
    return False

if __name__ == "__main__":
    ok = 0
    ensure_dir("data/desi_dr1_bao/BAO")
    for name, urls in CANDIDATES.items():
        out = os.path.join("data/desi_dr1_bao", name)
        if try_fetch(urls, out):
            ok += 1
    if ok == 0:
        print("\nNo BAO files fetched. You can manually drop a CSV with columns like:")
        print("  z, DM_over_r_d, DH_over_r_d, (optionally DV_over_r_d) and a matching covariance CSV.")
        raise SystemExit(1)
    print("\nFetch complete. Run scripts/check_data.py next to refresh the manifest.")
