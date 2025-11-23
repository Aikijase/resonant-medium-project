#!/usr/bin/env python3
"""
Write <stem>.override with a single metric and kind (default CHI2).

Usage:
  python3 tools/write_metric_override.py <stem> <value> [--kind CHI2]

Examples:
  python3 tools/write_metric_override.py outputs/phase8/joint_kappa0 302.2
  python3 tools/write_metric_override.py outputs/phase8/joint_A0 315.8 --kind CHI2
"""
import argparse, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="e.g. outputs/phase8/joint_kappa0")
    ap.add_argument("value", type=float, help="metric value (float)")
    ap.add_argument("--kind", default="CHI2", help="label for the metric (default CHI2)")
    args = ap.parse_args()

    out = f"{args.stem}.override"
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"metric: {args.value}\nkind: {args.kind.upper()}\n")
    print("WROTE", out)

if __name__ == "__main__":
    main()
