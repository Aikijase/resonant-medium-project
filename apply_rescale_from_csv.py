#!/usr/bin/env python3
"""
apply_rescale_from_csv.py
- Reads outputs/collapse_rescale_results.csv
- Updates data/registry.yaml by setting normalize.scale_omegahat per dataset
- Prints a summary of changes
"""

from pathlib import Path
import sys
import pandas as pd
import yaml

ROOT = Path.home() / "resonant-medium-project"
REG_PATH = ROOT / "data" / "registry.yaml"
CSV_PATH = ROOT / "outputs" / "collapse_rescale_results.csv"

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: missing {CSV_PATH}")
        sys.exit(1)
    if not REG_PATH.exists():
        print(f"ERROR: missing {REG_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if not {"dataset","scale_s"}.issubset(df.columns):
        print("ERROR: CSV must have columns: dataset, scale_s")
        sys.exit(1)

    reg = yaml.safe_load(open(REG_PATH, "r"))
    if "benchmarks" not in reg or not isinstance(reg["benchmarks"], dict):
        print("ERROR: registry.yaml missing 'benchmarks' section")
        sys.exit(1)

    changes = []
    for _, row in df.iterrows():
        name = str(row["dataset"])
        scale = float(row["scale_s"])
        item = reg["benchmarks"].get(name)
        if item is None:
            changes.append(f"[SKIP] {name} not found in registry.yaml")
            continue
        norm = item.setdefault("normalize", {})
        old = norm.get("scale_omegahat", None)
        norm["scale_omegahat"] = scale
        changes.append(f"[SET] {name}: scale_omegahat {old} → {scale:.9g}")

    yaml.safe_dump(reg, open(REG_PATH, "w"), sort_keys=False)
    print("\n".join(changes))
    print(f"\nWrote updates to {REG_PATH}")

if __name__ == "__main__":
    main()
