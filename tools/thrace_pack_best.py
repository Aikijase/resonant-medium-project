#!/usr/bin/env python3
"""
Package outputs/thrace_best/ into a shareable archive.

Creates:
  - outputs/thrace_best_bundle_<timestamp>.zip
  - outputs/thrace_best_bundle_<timestamp>.tar.gz
  - corresponding .sha256 files

Usage:
  PYTHONPATH=. python tools/thrace_pack_best.py
  # or custom dir:
  PYTHONPATH=. python tools/thrace_pack_best.py --best-dir outputs/thrace_best
"""
from __future__ import annotations
import argparse, hashlib, os, tarfile, zipfile, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_sha256(path: Path):
    digest = sha256_of(path)
    (path.with_suffix(path.suffix + ".sha256")).write_text(f"{digest}  {path.name}\n")
    print(f"[pack] sha256  {digest}  {path.name}")

def main():
    ap = argparse.ArgumentParser(description="Pack the best Thrace bundle.")
    ap.add_argument("--best-dir", default="outputs/thrace_best", help="Folder to package")
    args = ap.parse_args()

    best = ROOT / args.best_dir
    if not best.exists():
        raise SystemExit(f"[pack] not found: {best}")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_zip = ROOT / f"outputs/thrace_best_bundle_{ts}.zip"
    out_tgz = ROOT / f"outputs/thrace_best_bundle_{ts}.tar.gz"

    # ZIP
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(best.rglob("*")):
            if p.is_file():
                arc = Path("thrace_best") / p.relative_to(best)
                zf.write(p, arcname=str(arc))
    print(f"[pack] wrote {out_zip}")
    write_sha256(out_zip)

    # TAR.GZ
    with tarfile.open(out_tgz, "w:gz") as tf:
        tf.add(best, arcname="thrace_best")
    print(f"[pack] wrote {out_tgz}")
    write_sha256(out_tgz)

    # List contents (top level)
    print("\n[pack] contents:")
    for p in sorted(best.iterdir()):
        kind = "/" if p.is_dir() else ""
        print(" ", p.name + kind)

if __name__ == "__main__":
    main()
