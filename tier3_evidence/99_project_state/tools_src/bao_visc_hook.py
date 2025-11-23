#!/usr/bin/env python3
# tools/bao_visc_hook.py
#
# BAO runner hook for optional viscous-DM reweighting, driven by env vars.
# Defaults are SAFE for pure BAO (DM/DH/DV unaffected).
#
# Env vars:
#   DM_GRID=outputs/joint_visc_dm_growth_grid.csv   # required to compute the ratio
#   DM_KEFF=0.20                                    # k_eff in h/Mpc (default 0.20)
#   BAO_SCALE=0|1                                   # 1 to actually scale the model vector
#   BAO_SCALE_KINDS="FS8,BETA"                      # comma list (case-insensitive); DM/DH/DV not touched by default
#   BAO_VISCREPORT=outputs/bao_visc_report.csv      # optional path for a per-bin report
#
# Usage inside your runner (after building model vector m and having z_for_each_entry + kinds):
#   from tools.bao_visc_hook import maybe_scale_model_vector
#   m = np.array(m, float); y = np.array(y, float)
#   m, report_path = maybe_scale_model_vector(m, kinds, z_for_each_entry)
#
# If your BAO vector order matches zbao one-to-one, pass z_for_each_entry=zbao.

import os, csv, numpy as np
from typing import List, Tuple

# We reuse the fD ratio from the fs8 helper:
try:
    from tools.visc_fd_hook import visc_fd_ratio, _ensure_loaded
    _HAS_VISC = True
except Exception:
    _HAS_VISC = False

def _parse_kinds_list(s: str) -> List[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def _want_scale_kind(kind: str, scale_kinds: List[str]) -> bool:
    return kind.upper() in scale_kinds

def _compute_ratios_for_z(z_list: List[float], k_eff: float) -> List[float]:
    grid = _ensure_loaded()
    if grid is None:
        return [1.0]*len(z_list)
    return [float(visc_fd_ratio(float(z), k_eff, grid)) for z in z_list]

def maybe_scale_model_vector(m: np.ndarray,
                             kinds: List[str],
                             z_for_each_entry: List[float],
                             out_csv: str | None = None
                            ) -> Tuple[np.ndarray, str | None]:
    """
    If BAO_SCALE=1 and a viscous grid is configured, multiply m[i] by (fD)_visc/(fD)_LCDM
    for entries whose kind is in BAO_SCALE_KINDS (default: FS8,BETA). Returns (m_scaled, report_path).
    Always writes a report if BAO_VISCREPORT is set OR out_csv is given.
    """
    m = np.asarray(m, float)
    assert len(m) == len(kinds) == len(z_for_each_entry), "Length mismatch: m, kinds, z_for_each_entry must align."

    # Read env knobs
    scale_on = os.environ.get("BAO_SCALE", "0").strip() == "1"
    scale_kinds = _parse_kinds_list(os.environ.get("BAO_SCALE_KINDS", "FS8,BETA"))
    k_eff = float(os.environ.get("DM_KEFF", "0.20"))
    report_path = os.environ.get("BAO_VISCREPORT", None) or out_csv

    # If visc not available or scaling disabled, we still may dump a report with ratios for transparency.
    ratios = [1.0]*len(m)
    if _HAS_VISC:
        # require a DM grid to be loaded
        try:
            from tools.visc_fd_hook import _ensure_loaded as _ensure
            if _ensure() is not None:
                ratios = _compute_ratios_for_z(z_for_each_entry, k_eff)
        except Exception:
            pass

    m_scaled = m.copy()
    if scale_on and _HAS_VISC:
        for i, k in enumerate(kinds):
            if _want_scale_kind(k, scale_kinds):
                m_scaled[i] *= ratios[i]

    # optional report
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True) if os.path.dirname(report_path) else None
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx","z","kind","scale_applied","m_before","m_after"])
            for i, (z, k) in enumerate(zip(z_for_each_entry, kinds)):
                applied = (m_scaled[i]/m[i]) if (m[i] != 0.0) else (ratios[i] if scale_on else 1.0)
                w.writerow([i, float(z), str(k), applied, m[i], m_scaled[i]])
        print(f"[bao-visc] wrote report: {report_path}")

    return m_scaled, (report_path or None)
