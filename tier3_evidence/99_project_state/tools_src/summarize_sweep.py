#!/usr/bin/env python3
import csv, math, numpy as np
from pathlib import Path

path = Path("outputs/phase8/sweep_tau_kappa.csv")
if not path.exists():
    raise SystemExit(f"missing {path}")

def f(x):
    try:
        if x in ("", "None", None): return np.nan
        return float(x)
    except Exception:
        return np.nan

rows = []
with open(path) as fcsv:
    r = csv.DictReader(fcsv)
    for row in r:
        rows.append({
            "tau": f(row["tau"]),
            "kappa": f(row["kappa"]),
            "wmax": f(row["omega_at_max"]),
            "wq": f(row.get("quad_peak_omega", "")),
            "A": f(row.get("lorentz_A","")),
            "w0": f(row.get("lorentz_omega0","")),
            "gamma": f(row.get("lorentz_gamma","")),
            "Q": f(row.get("lorentz_Q","")),
            "status": row.get("status",""),
        })

T = sorted(set(r["tau"] for r in rows if not math.isnan(r["tau"])))
K = sorted(set(r["kappa"] for r in rows if not math.isnan(r["kappa"])))

W = np.full((len(T), len(K)), np.nan)
Qm = np.full_like(W, np.nan, dtype=float)
for rr in rows:
    if rr["status"]!="ok": continue
    i = T.index(rr["tau"]); j = K.index(rr["kappa"])
    W[i,j] = rr["w0"] if not math.isnan(rr["w0"]) else rr["wmax"]
    Qm[i,j] = rr["Q"]

# Global stats
w = W[np.isfinite(W)]
print(f"ω* finite count={w.size}, median={np.nanmedian(w):.6g}, min={np.nanmin(w):.6g}, max={np.nanmax(w):.6g}")
if np.nanmedian(w) > 0:
    drift = (np.nanmax(w) - np.nanmin(w)) / np.nanmedian(w) * 100
    print(f"global drift across grid ≈ {drift:.2f}%")

# Smoothness along τ (for each κ)
def smoothness_along_axis(M, axis):
    diffs = np.diff(M, axis=axis)
    return np.nanmedian(np.abs(diffs))

m_tau = smoothness_along_axis(W, axis=0)  # step in τ
m_kap = smoothness_along_axis(W, axis=1)  # step in κ
print(f"median |Δω*| per τ-step: {m_tau:.6g}")
print(f"median |Δω*| per κ-step: {m_kap:.6g}")

# Q-factor summary
q = Qm[np.isfinite(Qm)]
if q.size:
    print(f"Q finite count={q.size}, median Q={np.nanmedian(q):.3g}, min={np.nanmin(q):.3g}, max={np.nanmax(q):.3g}")
else:
    print("No finite Q (Lorentzian fits may have failed or flag disabled).")

# Spot any failed runs
fails = [rr for rr in rows if rr["status"]!="ok"]
if fails:
    print(f"\nFailures: {len(fails)} rows (showing up to 5):")
    for rr in fails[:5]:
        print(f"  τ={rr['tau']:.2f} κ={rr['kappa']:+.2f} status={rr['status']}")
