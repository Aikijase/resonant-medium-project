#!/usr/bin/env python3
import os, re, json

SEARCH_DIRS = [
    "data",
    os.path.expanduser("~/Downloads"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~"),
]

PATTERNS = {
    "planck2018": [
        re.compile(r"COM_Likelihood_Data-baseline_R3\.00", re.I),
        re.compile(r"(plik|Camspec|planck)", re.I),
    ],
    "pantheon_plus": [
        # Accept both "PantheonPlus..." and "Pantheon+..." variants
        re.compile(r"Pantheon(\+|Plus)SH0ES\.dat$", re.I),
        re.compile(r"Pantheon(\+|Plus)SH0ES_?STAT\+?SYS\.cov$", re.I),
        re.compile(r"Pantheon(\+|Plus)_Data", re.I),
    ],
    "desi_dr1_bao": [
        re.compile(r"\bbao\b", re.I),
        re.compile(r"\bDR1\b", re.I),
        re.compile(r"\bvac\b", re.I),
        re.compile(r"cosmo-params", re.I),
        re.compile(r"(measurements|covariance)", re.I),
    ],
}

def find_files():
    manifest = {k: [] for k in PATTERNS}
    for search_dir in SEARCH_DIRS:
        if not os.path.exists(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for f in files:
                full = os.path.join(root, f)
                for key, regexes in PATTERNS.items():
                    if any(rx.search(f) or rx.search(full) for rx in regexes):
                        manifest[key].append(full)
    # Deduplicate & sort
    for k in manifest:
        manifest[k] = sorted(set(manifest[k]))
    return manifest

if __name__ == "__main__":
    m = find_files()
    with open("data_manifest.json", "w") as fp:
        json.dump(m, fp, indent=2)
    for k, files in m.items():
        print(f"{k:15s}: {'FOUND' if files else 'MISSING'} ({len(files)} files)")
    print("\nManifest saved to data_manifest.json")
