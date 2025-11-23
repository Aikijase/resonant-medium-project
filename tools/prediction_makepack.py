#!/usr/bin/env python3
import json, math, hashlib, csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

def safe_read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return default

def extract_params(j, fallback):
    if j is None:
        return fallback
    g = j.get("params", j)
    return dict(
        A=float(g.get("A", fallback["A"])),
        tau_m=float(g.get("tau_m", g.get("tau", fallback["tau_m"]))),
        omega=float(g.get("omega", fallback["omega"])),
        phi=float(g.get("phi", fallback["phi"])),
        chi2=float(j.get("chi2", j.get("chi_sq", fallback["chi2"]))),
    )

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<16), b""):
            h.update(chunk)
    return h.hexdigest()

# Defaults if nothing is available
FALLBACK = {"A":1.0, "tau_m":1.0, "omega":1.0, "phi":0.0, "chi2":0.0}

# Load any available fits (P8 optional, P20/P21 preferred)
p8j  = safe_read_json(ROOT/"outputs/phase08/p08.fit.json")
p20j = safe_read_json(ROOT/"outputs/phase20/p20.fit.json")
p21j = safe_read_json(ROOT/"outputs/phase21/p21.fit.json")

p8  = extract_params(p8j,  FALLBACK)
p20 = extract_params(p20j, FALLBACK)
p21 = extract_params(p21j, FALLBACK)

# Choose primary params from P20 if present; else fall back
primary = p20 if p20j is not None else (p21 if p21j is not None else FALLBACK)
A, tau_m, omega, phi = primary["A"], primary["tau_m"], primary["omega"], primary["phi"]

# Guardband percent from CSV if available
gb_pct = 0.10
gb_csv = ROOT/"outputs/phase21/p21_guardbands.csv"
if gb_csv.exists():
    try:
        # Accept simple "key,value" or headerless pairs
        for line in gb_csv.read_text().splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) >= 2 and parts[0].lower().startswith("omega_pct_window"):
                gb_pct = abs(float(parts[1]))
                break
    except Exception:
        pass

# ------------------ Prediction 1: Lensing residual template -------------------
lmin, lmax, dl = 10, 1500, 10
ls = np.arange(lmin, lmax+1, dl)
t_scale = 3.0e-3  # fixed mapping for registration
ts = ls * t_scale
res = A * np.exp(-ts/max(tau_m, 1e-9)) * np.cos(omega*ts + phi)

rows1 = [(int(l), float(r)) for l, r in zip(ls, res)]
f1 = PRED_DIR/"lensing_residual_template.csv"
write_csv(f1, ["ell", "residual_template"], rows1)

# ------------------ Prediction 2: BAO phase tweak (proxy; ablation) ----------
z = np.linspace(0.1, 1.6, 151)
base = np.sin(6.0*z)
tweak = A * np.exp(-z/max(tau_m, 1e-9)) * np.cos(omega*z + phi) * 0.01
with_tau  = base + tweak
no_tau    = base
rows2 = [(float(zi), float(b), float(w), float(n)) for zi,b,w,n in zip(z, base, with_tau, no_tau)]
f2 = PRED_DIR/"bao_phase_tweak_proxy.csv"
write_csv(f2, ["z", "bao_base", "with_memory", "tau_to_zero"], rows2)

# ------------------ Markdown summary -----------------------------------------
md = PRED_DIR/"predictions_README.md"
md.write_text(f"""# Resonant Medium — Predictions (Lock Version)

This package declares two pre-registered, parameter-tied signatures derived from the Echo kernel.
Primary params sourced from Phase-20 fit if present (else Phase-21, else fallback).

## Kernel (effective)
A={A:.6g}, tau_m={tau_m:.6g}, omega={omega:.6g}, phi={phi:.6g}
chi2(P20)={p20['chi2']:.3f} ; chi2(P8)={p8['chi2']:.3f}

## P1: CMB/weak-lensing low-ω residual (template)
File: `lensing_residual_template.csv`
Definition: residual(l) = A·exp(-t(l)/tau_m)·cos(omega·t(l)+phi)
Mapping: t(l) = {t_scale} · l (fixed)
Guardband window: ±{gb_pct*100:.1f}% around omega

## P2: BAO phase tweak (proxy; ablation)
File: `bao_phase_tweak_proxy.csv`
with_memory = base(z) + 1%·A·exp(-z/tau_m)·cos(omega·z+phi)
Ablation: tau_to_zero ≡ base(z)

Notes: These are registration-ready templates, not a cosmology pipeline.
""")

# ------------------ Lockfile --------------------------------------------------
lock = PRED_DIR/"PRED_LOCK.SHA256"
with open(lock, "w") as f:
    for p in (f1, f2, md):
        f.write(f"{sha256_of(p)}  {p.name}\n")

print("[pred] wrote:", f1, f2, md, lock)
