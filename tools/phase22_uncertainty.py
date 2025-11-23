#!/usr/bin/env python3
"""
Phase-22 — Uncertainty Bands (drop-in)

Inputs (from Phase-21):
  - CSV OR JSON describing guardbands across omega^2 for one or more sync targets.

What this does:
  1) For each guardband point (omega2, Kphi_lo, Kphi_hi, target), compute centerline Kphi_mid.
  2) Estimate local slope d(si)/d(Kphi) at (omega2, Kphi_mid) via finite-difference.
  3) Bootstrap si at (omega2, Kphi_mid) across seeds to get sigma_si.
  4) Propagate to sigma_Kphi ≈ sigma_si / |dsi/dKphi|.
  5) Build 95% CIs for Kphi (centerline) and for band width.
  6) Write:
       - outputs/phase22/p22_uncertainty.csv
       - outputs/phase22/p22_bands.json
       - outputs/phase22/p22_manifest.json
  7) Print a small JSON summary to stdout.

Notes:
  - Robust loader supports many CSV/JSON shapes produced in Phase-21.
  - If slope is ~0, use a tiny epsilon to avoid blow-ups (and you’ll see large sigma_Kphi, which is expected).
"""

import argparse
import csv
import json
import math
import pathlib
import statistics
import subprocess
import sys
from collections import defaultdict

PY = sys.executable

# ------------------------------- Simulation harness -------------------------------

def run_si(omega2, Kphi, preset="neuron", kv=0.10, kx=0.20, eps=0.06, noise=0.01,
           adapt_every=15, steps=20000, burn_in=300, seed1=0, seed2=1, prefix="p22"):
    """
    Call tools/phase10_phasecouple_demo.py and return sync_index (float).
    Assumes that script prints a JSON payload with metrics.sync_index.
    """
    args = [
        PY, "tools/phase10_phasecouple_demo.py",
        "--preset", str(preset),
        "--omega2", str(omega2),
        "--kv", str(kv), "--kx", str(kx), "--Kphi", str(Kphi),
        "--eps", str(eps), "--adapt_every", str(adapt_every),
        "--noise", str(noise),
        "--steps", str(steps), "--burn_in", str(burn_in),
        "--seed1", str(seed1), "--seed2", str(seed2),
        "--prefix", str(prefix)
    ]
    p = subprocess.run(args, capture_output=True, text=True)
    p.check_returncode()
    out = json.loads(p.stdout)
    return float(out["metrics"]["sync_index"])

def finite_diff_slope(omega2, Kphi, dK=0.005, **kw):
    """Central finite-difference estimate of d(si)/d(Kphi)."""
    si_minus = run_si(omega2, Kphi - dK, **kw)
    si_plus  = run_si(omega2, Kphi + dK, **kw)
    slope = (si_plus - si_minus) / (2 * dK)
    return slope, si_minus, si_plus

def bootstrap_si(omega2, Kphi, n_boot=64, noise=0.01, seeds=(0, 1), **kw):
    """Bootstrap si by varying seeds deterministically."""
    vals = []
    s1, s2 = seeds
    for b in range(n_boot):
        si = run_si(omega2, Kphi, noise=noise, seed1=s1 + b, seed2=s2 + 17*b, **kw)
        vals.append(si)
    mu = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mu, sd, vals

# ------------------------------- Guardbands loader (robust) -------------------------------

_ALIASES = {
    "target":  ["target", "si_target", "si", "sync_index", "s_target"],
    "omega2":  ["omega2", "w2", "omega_sq", "omega^2"],
    "Kphi_lo": ["Kphi_lo", "Kphi_low", "K_lo", "Kphi_min", "Kphi_left"],
    "Kphi_hi": ["Kphi_hi", "Kphi_high", "K_hi", "Kphi_max", "Kphi_right"],
}

def _first_key(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return k
    return None

def _coerce_float(x, name):
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Could not parse '{name}' as float (value={x!r})") from e

def _normalize_row(row):
    """
    Map any alias set to canonical keys and coerce to float.
    Required canonical keys: target, omega2, Kphi_lo, Kphi_hi
    """
    out = {}
    for canon, aliases in _ALIASES.items():
        k = _first_key(row, aliases)
        if k is None:
            raise KeyError(
                f"Missing required column for '{canon}'. "
                f"Tried aliases: {aliases}. Got columns: {list(row.keys())}"
            )
        out[canon] = _coerce_float(row[k], canon)
    return out

def read_guardbands_csv(path_csv):
    rows = []
    with open(path_csv, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise KeyError("CSV appears to have no header row.")
        for raw in r:
            # skip empty lines
            if not any(v.strip() for v in raw.values() if isinstance(v, str)):
                continue
            rows.append(_normalize_row(raw))
    return rows

# JSON helpers

def _flatten_band_entry(entry):
    """
    Accept either:
      - per-target arrays: {"target": 0.95, "omega2":[...], "Kphi_lo":[...], "Kphi_hi":[...]}
      - per-point dicts:   {"target":0.95, "omega2":2.80, "Kphi_lo":0.8, "Kphi_hi":1.2}
    """
    rows = []
    if isinstance(entry, dict) and all(k in entry for k in ["omega2", "Kphi_lo", "Kphi_hi"]):
        if isinstance(entry["omega2"], list):
            tgt = entry.get("target") or entry.get("si_target") or entry.get("sync_index") or 0.95
            for w2, lo, hi in zip(entry["omega2"], entry["Kphi_lo"], entry["Kphi_hi"]):
                rows.append(_normalize_row({"target": tgt, "omega2": w2, "Kphi_lo": lo, "Kphi_hi": hi}))
        else:
            rows.append(_normalize_row(entry))
    return rows

def _deep_find_lists(obj):
    """Yield list objects found anywhere inside nested dict/list JSON."""
    if isinstance(obj, list):
        yield obj
        for x in obj:
            yield from _deep_find_lists(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _deep_find_lists(v)

def read_guardbands_json(path_json):
    with open(path_json) as f:
        data = json.load(f)

    # Case 1: top-level list
    if isinstance(data, list):
        rows = []
        for item in data:
            if isinstance(item, dict):
                rows.extend(_flatten_band_entry(item))
        if rows:
            return rows

    # Case 2: top-level dict with known containers
    if isinstance(data, dict):
        # 2a) object with "bands": [ {target, omega2:[...], Kphi_*:[...]} ... ]
        if "bands" in data and isinstance(data["bands"], list):
            rows = []
            for band in data["bands"]:
                rows.extend(_flatten_band_entry(band))
            if rows:
                return rows

        # 2b) common keys: rows / guardbands / data / points
        for key in ["rows", "guardbands", "data", "points", "entries"]:
            if key in data and isinstance(data[key], list):
                rows = []
                for item in data[key]:
                    rows.extend(_flatten_band_entry(item))
                if rows:
                    return rows

        # 2c) wide arrays per target:
        # {"targets":[...], "omega2":[...], "Kphi_lo":[[... per target ...]], "Kphi_hi":[[...]]}
        if all(k in data for k in ["targets", "omega2", "Kphi_lo", "Kphi_hi"]):
            targets = data["targets"]
            omega2  = data["omega2"]
            klo     = data["Kphi_lo"]
            khi     = data["Kphi_hi"]
            rows = []
            for t, arr_lo, arr_hi in zip(targets, klo, khi):
                for w2, lo, hi in zip(omega2, arr_lo, arr_hi):
                    rows.append(_normalize_row({"target": t, "omega2": w2, "Kphi_lo": lo, "Kphi_hi": hi}))
            if rows:
                return rows

        # 2d) flat arrays for a single target
        if all(k in data for k in ["omega2", "Kphi_lo", "Kphi_hi"]) and isinstance(data["omega2"], list):
            tgt = data.get("target", data.get("si_target", data.get("sync_index", 0.95)))
            rows = []
            for w2, lo, hi in zip(data["omega2"], data["Kphi_lo"], data["Kphi_hi"]):
                rows.append(_normalize_row({"target": tgt, "omega2": w2, "Kphi_lo": lo, "Kphi_hi": hi}))
            if rows:
                return rows

        # 2e) brute-scan any nested list for dict items we can normalize
        for lst in _deep_find_lists(data):
            rows = []
            ok = False
            for item in lst:
                if isinstance(item, dict):
                    try:
                        rows.extend(_flatten_band_entry(item))
                        ok = True
                    except Exception:
                        pass
            if ok and rows:
                return rows

    raise KeyError("JSON structure not recognized for guardbands.")

def read_guardbands(path_csv=None, path_json=None):
    """
    Try CSV first (if given), then JSON; bubble up the last error if both fail.
    """
    last_err = None
    if path_csv:
        try:
            return read_guardbands_csv(path_csv)
        except Exception as e:
            last_err = e
    if path_json:
        try:
            return read_guardbands_json(path_json)
        except Exception as e:
            last_err = e
    msg = "Failed to read guardbands."
    if last_err:
        msg += f" Last error: {last_err}"
    raise RuntimeError(msg)

# ------------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase-22: Uncertainty Bands (robust, drop-in)")
    ap.add_argument("--guardbands-csv", help="Phase-21 CSV (any header variant)")
    ap.add_argument("--guardbands-json", help="Phase-21 JSON (flexible shapes)")
    ap.add_argument("--outdir", default="outputs/phase22")
    ap.add_argument("--preset", default="neuron")
    ap.add_argument("--kv", type=float, default=0.10)
    ap.add_argument("--kx", type=float, default=0.20)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=64)
    ap.add_argument("--dK", type=float, default=0.005, help="Finite-diff step for slope")
    ap.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI level (0.05 -> 95%)")
    ap.add_argument("--prefix", default="p22")
    args = ap.parse_args()

    if not args.guardbands_csv and not args.guardbands_json:
        raise SystemExit("Provide --guardbands-csv or --guardbands-json (from Phase-21).")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load guardbands
    gb = read_guardbands(args.guardbands_csv, args.guardbands_json)
    if not gb:
        raise SystemExit("Guardbands input is empty.")

    rows_unc = []
    by_target = defaultdict(list)

    for g in gb:
        target = g["target"]
        w2     = g["omega2"]
        K_lo   = g["Kphi_lo"]
        K_hi   = g["Kphi_hi"]
        K_mid  = 0.5 * (K_lo + K_hi)

        # 1) local slope at centerline
        slope, si_minus, si_plus = finite_diff_slope(
            w2, K_mid,
            dK=args.dK,
            preset=args.preset, kv=args.kv, kx=args.kx, eps=args.eps,
            noise=args.noise, adapt_every=15, steps=args.steps, burn_in=args.burn_in,
            seed1=0, seed2=1, prefix=f"{args.prefix}_slope_w{w2:.3f}_K{K_mid:.3f}"
        )

        # 2) bootstrap si at centerline
        mu_si, sd_si, _ = bootstrap_si(
            w2, K_mid, n_boot=args.n_boot, noise=args.noise,
            preset=args.preset, kv=args.kv, kx=args.kx, eps=args.eps,
            adapt_every=15, steps=args.steps, burn_in=args.burn_in, seeds=(2, 3),
            prefix=f"{args.prefix}_boot_w{w2:.3f}_K{K_mid:.3f}"
        )

        # 3) sigma propagation
        slope_abs = abs(slope) if abs(slope) > 1e-12 else 1e-12
        sigma_K   = sd_si / slope_abs
        z95       = 1.96 if abs(args.alpha - 0.05) < 1e-9 else 1.96  # simple
        dK_CI     = z95 * sigma_K

        band_width = (K_hi - K_lo)

        row = {
            "target": target,
            "omega2": w2,
            "Kphi_lo": K_lo,
            "Kphi_hi": K_hi,
            "Kphi_mid": K_mid,
            "si_mid": mu_si,
            "si_sd": sd_si,
            "dsi_dKphi": slope,
            "sigma_Kphi": sigma_K,
            "Kphi_CI_lo": K_mid - dK_CI,
            "Kphi_CI_hi": K_mid + dK_CI,
            "band_width": band_width,
            "band_width_CI_lo": max(0.0, band_width - 2*dK_CI),
            "band_width_CI_hi": band_width + 2*dK_CI,
            "n_boot": args.n_boot
        }
        rows_unc.append(row)
        by_target[target].append(row)

    # Write CSV
    csv_path = outdir / "p22_uncertainty.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_unc[0].keys()))
        w.writeheader()
        w.writerows(rows_unc)

    # Aggregate per target for ribbons
    bands = []
    for target, rows in sorted(by_target.items()):
        rows = sorted(rows, key=lambda r: r["omega2"])
        bands.append({
            "target": target,
            "omega2": [r["omega2"] for r in rows],
            "Kphi_mid": [r["Kphi_mid"] for r in rows],
            "Kphi_CI_lo": [r["Kphi_CI_lo"] for r in rows],
            "Kphi_CI_hi": [r["Kphi_CI_hi"] for r in rows],
        })

    # Manifest
    manifest = {
        "inputs": {
            "guardbands_csv": args.guardbands_csv,
            "guardbands_json": args.guardbands_json,
            "preset": args.preset, "kv": args.kv, "kx": args.kx, "eps": args.eps,
            "noise": args.noise, "steps": args.steps, "burn_in": args.burn_in,
            "n_boot": args.n_boot, "dK": args.dK, "alpha": args.alpha,
        },
        "outputs": {
            "uncertainty_csv": str(csv_path),
            "bands_json": str(outdir / "p22_bands.json"),
        },
        "notes": "σ_Kphi via linear error propagation; robust guardband loader for Phase-21 shapes."
    }

    with open(outdir / "p22_bands.json", "w") as f:
        json.dump(bands, f, indent=2)
    with open(outdir / "p22_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Final stdout summary
    print(json.dumps({
        "status": "ok",
        "csv": str(csv_path),
        "bands_json": str(outdir / "p22_bands.json"),
        "manifest": str(outdir / "p22_manifest.json"),
        "n_rows": len(rows_unc),
        "targets": sorted(set([r["target"] for r in rows_unc])),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
