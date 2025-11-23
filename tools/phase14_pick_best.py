#!/usr/bin/env python3
"""
Phase 14 — Pick Best Locking Cells

Scans one or more p14_*.csv lockmap outputs and writes a top-N CSV of
the strongest sync-index (Si) cells, plus prints the top few to stdout.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

def read_rows(csv_path: Path):
    rows=[]
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                si = float(r.get("Si","").strip()) if r.get("Si","").strip() else float("nan")
            except Exception:
                si = float("nan")
            try:
                o = float(r.get("omega2",""))
                k = float(r.get("Kphi",""))
            except Exception:
                continue
            rows.append((o, k, si))
    return rows

def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 14 — pick top-N lock cells")
    ap.add_argument("--glob", default="outputs/phase14/p14_*.csv", help="CSV glob to search")
    ap.add_argument("--topn", type=int, default=10, help="How many rows to keep")
    ap.add_argument("--min-si", type=float, default=0.0, help="Keep only rows with Si >= min")
    ap.add_argument("--out", type=Path, default=Path("outputs/phase14/p14_top.csv"))
    args = ap.parse_args(argv)

    files = sorted(Path().glob(args.glob))
    if not files:
        print(f"[pick] no files for glob: {args.glob}")
        return 2

    all_rows=[]
    for fp in files:
        for (o,k,si) in read_rows(fp):
            if not math.isnan(si) and si >= args.min_si:
                all_rows.append((si,o,k,fp.name))

    all_rows.sort(key=lambda x: x[0], reverse=True)
    keep = all_rows[:args.topn]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Si","omega2","Kphi","source"])
        for si,o,k,src in keep:
            wr.writerow([f"{si:.6f}", f"{o:.6f}", f"{k:.6f}", src])

    print(f"[pick] wrote: {args.out} (top {len(keep)} rows)")
    for si,o,k,src in keep[:min(5,len(keep))]:
        print(f"  Si={si:.4f} @ omega2={o:.3f}, Kphi={k:.3f}  ({src})")
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
