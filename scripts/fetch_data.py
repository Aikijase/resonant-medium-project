#!/usr/bin/env python3
import os, urllib.request

TARGETS = {
    "planck2018": [
        ("https://esdcdoi.esac.esa.int/doi/html/data/astronomy/planck/COM_Likelihood_Data-baseline_R3.00.tar.gz",
         "data/planck2018/COM_Likelihood_Data-baseline_R3.00.tar.gz"),
    ],
    "pantheon_plus": [
        # Distance vector + full STAT+SYS covariance (note the %2B encodes '+')
        ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
         "data/pantheon_plus/Pantheon+SH0ES.dat"),
        ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
         "data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.cov"),
        # (Optional) stat-only covariance:
        # ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STATONLY.cov",
        #  "data/pantheon_plus/Pantheon+SH0ES_STATONLY.cov"),
    ],
    "desi_dr1_bao": [
        # Pointer file so the checker knows DR1 BAO is present; you already have other DR1 BAO files FOUND.
        ("https://data.desi.lbl.gov/doc/releases/dr1/vac/bao-cosmo-params/",
         "data/desi_dr1_bao/INDEX.txt"),
    ],
}

def ensure_dir(path): os.makedirs(os.path.dirname(path), exist_ok=True)

def download(url, out):
    ensure_dir(out)
    if os.path.exists(out):
        print(f"Skip (exists): {out}")
        return
    print(f"Downloading {url} -> {out}")
    urllib.request.urlretrieve(url, out)
    print("Done.")

if __name__ == "__main__":
    for group, items in TARGETS.items():
        for url, rel in items:
            try:
                download(url, rel)
            except Exception as e:
                print(f"[WARN] Could not fetch {url}: {e}")
    print("Fetch complete. Next: run scripts/check_data.py")
