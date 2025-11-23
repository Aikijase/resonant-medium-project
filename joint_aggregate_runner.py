#!/usr/bin/env python3
"""
Joint Aggregate Runner — makes a guardable joint χ² from existing per-dataset JSONs.

Usage examples
--------------
# Minimal: point at your existing per-dataset summary JSONs
python3 joint_aggregate_runner.py \
  --in outputs/sn_fit_zhd_summary.json outputs/bao_fit_active_ln2_diag.json \
  --out outputs/joint_FROM_SUMMARIES.json

# With guard checks (friendly mode)
JOINT_NO_EXIT=1 python3 joint_aggregate_runner.py \
  --in outputs/sn_fit_zhd_summary.json outputs/bao_fit_active_ln2_diag.json \
  --out outputs/joint_FROM_SUMMARIES.json \
  --assert-chi2-between 600 900 \
  --assert-chi2-ndof-max 1.2
"""

from __future__ import annotations
import os, sys, json, argparse, re
from pathlib import Path
from typing import Tuple, Any, Dict, List

NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

# ---------- Helpers ----------
def read_json(p: Path) -> dict:
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _dig(d: dict, keys: List[str]):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None

def _extract_metrics(data: dict) -> Tuple[float | None, int | float | None]:
    """Tolerant extraction of chi2 and ndof from many common shapes."""
    chi2 = _dig(data, ["chi2","chi2_min","chi2_total","S","S_best"])
    ndof = _dig(data, ["ndof","ndf","dof","nu","n_dof","ndof_total","dof_total"])

    # look inside common containers
    for box in ("result","results","metrics","fit","summary"):
        v = data.get(box)
        if isinstance(v, dict):
            if chi2 is None:
                chi2 = _dig(v, ["chi2","chi2_min","chi2_total","S","S_best"])
            if ndof is None:
                ndof = _dig(v, ["ndof","ndf","dof","nu","n_dof","ndof_total","dof_total"])

    # fallback: ndof = N - k
    if ndof in (None, 0):
        N = _dig(data, ["nobs","N","ndata","n_data"])
        k = _dig(data, ["k_params","nparams","k","n_free","n_par"])
        if isinstance(N,(int,float)) and isinstance(k,(int,float)):
            ndof = int(N - k)
    return chi2, ndof

def normalize_between_tokens(argv: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--assert-chi2-between="):
            val = tok.split("=", 1)[1]
            if (i + 1) < len(argv) and ("," not in val) and (":" not in val) and NUM_RE.match(val) and NUM_RE.match(argv[i+1]):
                out.append(f"--assert-chi2-between={val},{argv[i+1]}")
                i += 2
                continue
        out.append(tok)
        i += 1
    return out

class BetweenAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            values = [values]
        try:
            if len(values) == 1:
                s = values[0].lstrip("=").replace(":", ",")
                a, b = [float(x) for x in s.split(",")]
            else:
                a, b = float(values[0]), float(values[1])
            setattr(namespace, self.dest, (a, b))
        except Exception:
            setattr(namespace, self.dest, None)
            if not hasattr(namespace, "_arg_errors"):
                namespace._arg_errors = []  # type: ignore[attr-defined]
            namespace._arg_errors.append("--assert-chi2-between: expected two numbers (e.g., 600 900 or 600,900)")  # type: ignore[attr-defined]

# ---------- CLI / Main ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inputs", nargs="+", required=True,
                   help="List of per-dataset summary JSONs to aggregate (each should contain chi2 and dof/ndof).")
    p.add_argument("--out", type=Path, required=True, help="Output joint JSON path to write.")
    # Optional guard assertions (same flavor as your guard)
    p.add_argument("--assert-chi2-max", type=float)
    p.add_argument("--assert-chi2-ndof-max", type=float)
    p.add_argument("--assert-chi2-between", nargs="+", action=BetweenAction)
    p.add_argument("--no-exit", action="store_true")
    return p

def guard_and_print(chi2: float, ndof: float, between, chi2_max, ndof_max) -> int:
    ok = True
    low, high = between if between else (None, None)
    if chi2_max is not None and chi2 > chi2_max:
        print(f"ASSERT   failed: chi2={chi2:.3f} > max={chi2_max}")
        ok = False
    if between is not None and not (low <= chi2 <= high):
        print(f"ASSERT   failed: chi2={chi2:.3f} not in [{low},{high}]")
        ok = False
    r = chi2 / ndof if ndof else float("inf")
    if ndof_max is not None and r > ndof_max:
        print(f"ASSERT   failed: chi2/ndof={r:.3f} > max={ndof_max}")
        ok = False
    if ok:
        print(f"ASSERT   ok  chi2={chi2:.3f}  chi2/ndof={r:.3f}")
        return 0
    return 2

def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = normalize_between_tokens(list(argv))
    args = build_parser().parse_args(argv)

    total_chi2 = 0.0
    total_ndof = 0.0
    parts: List[Dict[str, Any]] = []

    for p in args.inputs:
        pj = Path(p)
        d = read_json(pj)
        if not d:
            print(f"[warn] could not read JSON: {p}")
            continue
        chi2, ndof = _extract_metrics(d)
        if chi2 is None or ndof in (None, 0):
            print(f"[warn] missing chi2/ndof in {p} (skipping)")
            continue
        parts.append({"path": str(pj), "chi2": float(chi2), "ndof": float(ndof)})
        total_chi2 += float(chi2)
        total_ndof += float(ndof)

    if total_ndof == 0:
        print("ASSERT   failed: no valid inputs with chi2+ndof")
        rc = 2
    else:
        # write joint JSON
        out = {
            "chi2": total_chi2,
            "ndof": int(total_ndof),
            "parts": parts,
            "note": "Aggregated from per-dataset summaries (no refit)."
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        # run guard if flags are present
        if any(getattr(args, n) is not None for n in ("assert_chi2_max", "assert_chi2_ndof_max", "assert_chi2_between")):
            rc = guard_and_print(
                total_chi2, total_ndof,
                getattr(args, "assert_chi2_between", None),
                getattr(args, "assert_chi2_max", None),
                getattr(args, "assert_chi2_ndof_max", None),
            )
        else:
            print(f"[info] wrote joint JSON to {args.out}")
            rc = 0

    if args.no_exit or os.environ.get("JOINT_NO_EXIT") == "1":
        print(f"[info] joint_aggregate_runner completed with exit code {rc}")
        return rc
    raise SystemExit(rc)

if __name__ == "__main__":
    main()
