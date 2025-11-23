#!/usr/bin/env python3
"""
Phase Autopack (Phase-agnostic)

Scans an outputs/<phaseX> directory, builds CSV+JSON manifests with hashes
and metadata, writes a README, and optionally produces a ZIP evidence pack.

Design:
- Single-file, no external deps (stdlib only).
- Predictable outputs under <root>/manifest/.
- Re-runnable and idempotent.
- Label support to name manifest files (e.g., --label phase14).

Outputs:
- <root>/manifest/<label>_manifest.csv
- <root>/manifest/<label>_manifest.json
- <root>/manifest/README_MANIFEST.txt
- (optional) <root>/<zip> or ABSOLUTE_PATH.zip

Usage examples:
  # Manifests only
  python3 tools/phase13_autopack.py --root outputs/phase14 --label phase14

  # Manifests + ZIP to ChromeOS storage
  python3 tools/phase13_autopack.py \
    --root outputs/phase14 \
    --label phase14 \
    --bundle \
    --bundle-name "/mnt/chromeos/MyFiles/Downloads/resonant-archives/phase14_bundle.zip"

  # Filter by extensions
  python3 tools/phase13_autopack.py \
    --root outputs/phase13 \
    --label phase13 \
    --include-exts .csv .json .png .pdf
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ------------------------------
# Utilities
# ------------------------------

TIMESTAMP_PATTERNS = [
    re.compile(r"(?P<date>\d{8})[-_](?P<time>\d{6})"),            # 20251103-142230 or 20251103_142230
    re.compile(r"(?P<date>\d{8})(?P<time>\d{6})"),                # 20251103142230
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[T_\-]?(?P<time>\d{2}:\d{2}:\d{2})"),  # 2025-11-03T14:22:30
]

def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def infer_tag_and_prefix(rel: Path) -> Tuple[str, str]:
    """Infer (tag, prefix) from relative path."""
    parts = rel.parts
    tag = parts[0] if parts else ''
    stem = rel.stem
    m = re.match(r"([A-Za-z0-9\-]+)_", stem)
    prefix = m.group(1) if m else stem
    return tag, prefix

def extract_run_id(name: str) -> str:
    for rx in TIMESTAMP_PATTERNS:
        m = rx.search(name)
        if m:
            date = re.sub(r"[^0-9]", "", m.group('date'))
            time = re.sub(r"[^0-9]", "", m.group('time'))
            return f"{date}-{time}"
    return ""

def guess_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {'.csv'}: return 'csv'
    if ext in {'.json'}: return 'json'
    if ext in {'.png', '.jpg', '.jpeg', '.gif', '.svg'}: return 'image'
    if ext in {'.pdf'}: return 'pdf'
    if ext in {'.txt', '.log'}: return 'text'
    return ext.lstrip('.') or 'file'

@dataclass
class Row:
    relpath: str
    bytes: int
    mtime_iso: str
    kind: str
    sha256: str
    tag: str
    prefix: str
    run_id: str
    def to_csv_row(self) -> List[str]:
        return [self.relpath, str(self.bytes), self.mtime_iso, self.kind,
                self.sha256, self.tag, self.prefix, self.run_id]

CSV_HEADER = ['relpath', 'bytes', 'mtime_iso', 'kind', 'sha256', 'tag', 'prefix', 'run_id']

# ------------------------------
# Scanner
# ------------------------------

def iter_files(root: Path,
               include_exts: Tuple[str, ...],
               exclude_globs: Tuple[str, ...],
               follow_symlinks: bool = False) -> Iterable[Path]:
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if not follow_symlinks and p.is_symlink():
            continue
        rel = p.relative_to(root)
        if any(fnmatch.fnmatch(str(rel), pat) for pat in exclude_globs):
            continue
        if include_exts and p.suffix.lower() not in include_exts:
            continue
        yield p

def build_manifest(root: Path,
                   include_exts: Tuple[str, ...],
                   exclude_globs: Tuple[str, ...]) -> List[Row]:
    rows: List[Row] = []
    for path in iter_files(root, include_exts, exclude_globs):
        stat = path.stat()
        rel = path.relative_to(root)
        tag, prefix = infer_tag_and_prefix(rel)
        run_id = extract_run_id(path.name)
        kind = guess_kind(path)
        sha = sha256_of(path)
        mtime_iso = dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds')
        rows.append(Row(str(rel), stat.st_size, mtime_iso, kind, sha, tag, prefix, run_id))
    rows.sort(key=lambda r: (r.kind, r.relpath))
    return rows

# ------------------------------
# Writers
# ------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_csv(rows: List[Row], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow(r.to_csv_row())

def write_json(rows: List[Row], out_json: Path) -> None:
    ensure_dir(out_json.parent)
    with out_json.open('w') as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

def write_readme(root: Path, out_txt: Path, args: argparse.Namespace, nrows: int) -> None:
    ensure_dir(out_txt.parent)
    now = dt.datetime.now().isoformat(timespec='seconds')
    lines = []
    lines.append("Phase Autopack — Manifest Readme\n")
    lines.append(f"Generated: {now}\n")
    lines.append(f"Root: {root}\n")
    lines.append(f"Label: {args.label or Path(root).name}\n")
    lines.append(f"Files indexed: {nrows}\n\n")
    lines.append("Flags:\n")
    lines.append(f"  include_exts: {', '.join(args.include_exts) if args.include_exts else '(all)'}\n")
    lines.append(f"  exclude_globs: {', '.join(args.exclude_globs) if args.exclude_globs else '(none)'}\n")
    lines.append(f"  bundle: {'yes' if args.bundle else 'no'}\n")
    if args.notes:
        lines.append("\nNotes:\n")
        lines.append(args.notes.strip() + "\n")
    out_txt.write_text(''.join(lines))

def make_bundle(root: Path, rows: List[Row], out_zip: Path, extra_files: Iterable[Path] = ()) -> None:
    ensure_dir(out_zip.parent)
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            z.write(root / r.relpath, arcname=r.relpath)
        for ef in extra_files:
            arc = Path('manifest') / ef.name
            z.write(ef, arcname=str(arc))

# ------------------------------
# CLI
# ------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Phase Autopack')
    p.add_argument('--root', type=Path, default=Path('outputs/phase13'),
                   help='Root directory to scan (default: outputs/phase13)')
    p.add_argument('--include-exts', nargs='*',
                   default=['.csv', '.json', '.png', '.jpg', '.jpeg', '.pdf', '.txt'],
                   help='Extensions to include. Empty = all files.')
    p.add_argument('--exclude-globs', nargs='*',
                   default=['manifest/*', '**/tmp/*', '**/.ipynb_checkpoints/*'],
                   help='Glob patterns to exclude from scan')
    p.add_argument('--bundle', action='store_true', help='Create a ZIP evidence pack')
    p.add_argument('--bundle-name', default='', help='Custom zip name (no path if relative). Absolute path respected.')
    p.add_argument('--label', default='', help='Label for manifest filenames (default: root folder name)')
    p.add_argument('--notes', default='', help='Notes to embed in README_MANIFEST.txt')
    p.add_argument('--follow-symlinks', action='store_true', help='Follow symlinks (off by default)')
    p.add_argument('--dry-run', action='store_true', help='Scan and print counts, but do not write files')
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root: Path = args.root
    if not root.exists():
        print(f"[autopack] root does not exist: {root}", file=sys.stderr)
        return 2

    include_exts = tuple((e.lower() if e.startswith('.') else f'.{e.lower()}') for e in (args.include_exts or ()))
    exclude_globs = tuple(args.exclude_globs or ())

    print(f"[autopack] scanning: {root}")
    rows = build_manifest(root, include_exts, exclude_globs)
    print(f"[autopack] files indexed: {len(rows)}")

    if args.dry_run:
        return 0

    # Label for filenames
    label = args.label.strip() or Path(root).name or "phase"

    manifest_dir = root / 'manifest'
    out_csv  = manifest_dir / f'{label}_manifest.csv'
    out_json = manifest_dir / f'{label}_manifest.json'
    out_txt  = manifest_dir / 'README_MANIFEST.txt'

    write_csv(rows, out_csv)
    write_json(rows, out_json)
    write_readme(root, out_txt, args, len(rows))

    print(f"[autopack] wrote: {out_csv}")
    print(f"[autopack] wrote: {out_json}")
    print(f"[autopack] wrote: {out_txt}")

    if args.bundle:
        ts = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        zip_name = args.bundle_name.strip() or f'{label}_evidence_pack_{ts}.zip'
        out_zip = Path(zip_name) if os.path.isabs(zip_name) else (root / zip_name)
        try:
            make_bundle(root, rows, out_zip, extra_files=[out_csv, out_json, out_txt])
            print(f"[autopack] wrote: {out_zip}")
        except OSError as e:
            if getattr(e, "errno", None) == 28:
                print("[autopack] ERROR: No space left on device during ZIP. "
                      "Free space, filter with --include-exts, or pass an absolute --bundle-name to another drive.",
                      file=sys.stderr)
            else:
                raise

    print("[autopack] done.")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
