#!/usr/bin/env python3
# Phase-12 summariser (robust aliases + NaN-safe plots)
# - Ingests multiple CSVs if present
# - Normalises column names (α/λ2/R/K/span/ivar/seed/noise/br)
# - Computes H with fallbacks (R*K, or R-only, or K-only)
# - Writes: phase12_summary.csv, phase12_alpha_star.csv, phase12_report_card.txt
# - Plots: R/K/span vs α (if present), plus H vs α (+ by-seed)
# - Dumps aggregation for debug: phase12_H_agg_by_alpha.csv

import sys, json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_FILES = [
    "outputs/phase12/data/multichord_wide.csv",
    "outputs/phase12/data/multichord_random_k.csv",
    "outputs/phase12/data/multichord_alpha_sweep.csv",
]

OUTDIR  = Path("outputs/phase12")
PLOTDIR = OUTDIR / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
def read_any(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

dfs = [read_any(f) for f in IN_FILES if Path(f).exists()]
if not dfs:
    print("No input CSVs found. Expected one of:", IN_FILES, file=sys.stderr)
    sys.exit(1)
raw = pd.concat(dfs, ignore_index=True)

# ---------- Alias normalisation ----------
def norm(s: str) -> str:
    return (
        s.strip()
         .lower()
         .replace('λ','lambda')
         .replace('α','alpha')
         .replace(' ','')
         .replace('_','')
    )

colmap = {norm(c): c for c in raw.columns}

def pick(*cands, default=None):
    for c in cands:
        k = norm(c)
        if k in colmap:
            return colmap[k]
    return default

COL_BR    = pick('br','branch','branchratio')
COL_A     = pick('α','alpha','a')
COL_SEED  = pick('seed')
COL_NOISE = pick('noise','sigma')
COL_R     = pick('R','r','resonance','syncindex','si','sync','S')
COL_K     = pick('K','k','coherence','kindex','keff','kappa_eff')
COL_SPAN  = pick('span','bandwidth','width')
COL_IVAR  = pick('ivar','intervar','inter_variance','variance')
COL_L2    = pick('λ2','lambda2','l2','eig2')

cols_order = [c for c in [COL_BR,COL_A,COL_SEED,COL_NOISE,COL_R,COL_K,COL_SPAN,COL_IVAR,COL_L2] if c]
df = raw[cols_order].copy()

rename_to = {}
if COL_BR:    rename_to[COL_BR]    = 'br'
if COL_A:     rename_to[COL_A]     = 'α'
if COL_SEED:  rename_to[COL_SEED]  = 'seed'
if COL_NOISE: rename_to[COL_NOISE] = 'noise'
if COL_R:     rename_to[COL_R]     = 'R'
if COL_K:     rename_to[COL_K]     = 'K'
if COL_SPAN:  rename_to[COL_SPAN]  = 'span'
if COL_IVAR:  rename_to[COL_IVAR]  = 'ivar'
if COL_L2:    rename_to[COL_L2]    = 'λ2'
df.rename(columns=rename_to, inplace=True)

# Ensure numeric
for c in ['br','α','seed','noise','R','K','span','ivar','λ2']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ---------- Harmony score with fallbacks ----------
eps = 1e-9
target_span = df['span'].median() if 'span' in df.columns and df['span'].notna().any() else None
def penal_span(s):
    if target_span is None or pd.isna(s): return 0.0
    return max(0.0, s - target_span)

def denom_row(row):
    base = 1.0
    if 'ivar' in df.columns and not pd.isna(row.get('ivar', np.nan)): base += row['ivar']
    if 'span' in df.columns and not pd.isna(row.get('span', np.nan)): base += penal_span(row['span'])
    return base + eps

def harmony(row):
    hasR = ('R' in df.columns) and np.isfinite(row.get('R', np.nan))
    hasK = ('K' in df.columns) and np.isfinite(row.get('K', np.nan))
    if hasR and hasK: num = row['R'] * row['K']
    elif hasR:        num = row['R']
    elif hasK:        num = row['K']
    else:             return np.nan
    return num / denom_row(row)

df['H'] = df.apply(harmony, axis=1)

# ---------- Save tidy summary ----------
sort_keys = [c for c in ['br','α','seed'] if c in df.columns]
if sort_keys: df.sort_values(sort_keys, inplace=True, na_position='last')
OUTCSV = OUTDIR / "phase12_summary.csv"
df.to_csv(OUTCSV, index=False)

# ---------- Alpha* ----------
alpha_star_df = pd.DataFrame(columns=['br','alpha_star','H_star'])
if 'α' in df.columns:
    groups = (['br','α'] if 'br' in df.columns else ['α'])
    med = df.groupby(groups)['H'].median().reset_index()
    if not med.empty:
        if 'br' in med.columns:
            alpha_star_df = (
                med.sort_values(['br','H'], ascending=[True, False])
                   .groupby('br').head(1)
                   .rename(columns={'α':'alpha_star','H':'H_star'})[['br','alpha_star','H_star']]
            )
        else:
            best = med.sort_values('H', ascending=False).head(1)
            if not best.empty:
                alpha_star_df = pd.DataFrame([{
                    'br': -1, 'alpha_star': best.iloc[0]['α'], 'H_star': best.iloc[0]['H']
                }])
alpha_star_path = OUTDIR / "phase12_alpha_star.csv"
if not alpha_star_df.empty:
    alpha_star_df.to_csv(alpha_star_path, index=False)

# ---------- Report card ----------
lines = ["Phase 12 — Report Card", ""]
if not alpha_star_df.empty:
    lines += ["br   α*     H*"]
    for _,r in alpha_star_df.iterrows():
        br = int(r['br']) if np.isfinite(r['br']) else -1
        lines.append(f"{br:<3}  {r['alpha_star']}   {r['H_star']:.3f}")
    lines.append("")
else:
    lines += [
        "No α* extracted (missing α or H).",
        "Tip: ensure CSV has α (alpha) and at least one of R or K (optional).",
        ""
    ]

# Seed stability (if possible)
if 'seed' in df.columns and (set(df.columns) & {'R','K','span','ivar'}):
    agg_cols = [c for c in ['R','K','span','ivar'] if c in df.columns]
    by = [c for c in ['br','α'] if c in df.columns]
    if by and agg_cols:
        stab = df.groupby(by)[agg_cols].agg(['mean','std']).reset_index()
        lines.append("Seed-stability (mean±std):")
        for _,row in stab.iterrows():
            tag = " ".join([f"{by[i]}={row[(by[i],'')]}" for i in range(len(by))])
            parts = [f"{m}={row[(m,'mean')]:.3f}±{row[(m,'std')]:.3f}" for m in agg_cols]
            lines.append(f"  {tag}: " + "  ".join(parts))

REPORT = OUTDIR / "phase12_report_card.txt"
with open(REPORT,'w') as f: f.write("\n".join(lines))

# ---------- Plot helpers (NaN-safe) ----------
def _safe_errorbar(x, y, yerr):
    if yerr is None:
        plt.plot(x, y, '-o'); return
    yerr = np.nan_to_num(np.asarray(yerr), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(yerr).any() or (yerr == 0).all():
        plt.plot(x, y, '-o')
    else:
        plt.errorbar(x, y, yerr=yerr, fmt='-o')

def plot_metric_vs_alpha(metric):
    if metric not in df.columns or 'α' not in df.columns: return
    tmp = df[['α',metric]].dropna().copy()
    tmp['α'] = pd.to_numeric(tmp['α'], errors='coerce')
    tmp[metric] = pd.to_numeric(tmp[metric], errors='coerce')
    tmp = tmp.dropna().sort_values('α')
    if tmp.empty: return
    piv = tmp.groupby('α')[metric].agg(['mean','std']).reset_index().sort_values('α')
    if piv.empty: return
    x = piv['α'].to_numpy(); y = piv['mean'].to_numpy()
    yerr = piv['std'].to_numpy() if 'std' in piv.columns else None
    plt.figure()
    _safe_errorbar(x, y, yerr)
    plt.title(f'{metric} vs α')
    plt.xlabel('α'); plt.ylabel(metric); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTDIR / f"{metric}_vs_alpha.png", dpi=160)
    plt.close()

# R/K/span if present
for m in ['R','K','span']:
    plot_metric_vs_alpha(m)

# H (always if present)
if 'H' in df.columns and 'α' in df.columns:
    # mean±std
    tmp = df[['α','H']].dropna().copy()
    tmp['α'] = pd.to_numeric(tmp['α'], errors='coerce')
    tmp['H'] = pd.to_numeric(tmp['H'], errors='coerce')
    tmp = tmp.dropna().sort_values('α')
    if not tmp.empty:
        piv = tmp.groupby('α')['H'].agg(['mean','std']).reset_index().sort_values('α')
        piv.to_csv(OUTDIR / "phase12_H_agg_by_alpha.csv", index=False)  # debug
        x = piv['α'].to_numpy(); y = piv['mean'].to_numpy()
        yerr = piv['std'].to_numpy() if 'std' in piv.columns else None
        plt.figure()
        _safe_errorbar(x, y, yerr)
        plt.title('H vs α')
        plt.xlabel('α'); plt.ylabel('H'); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTDIR / "H_vs_alpha.png", dpi=160)
        plt.close()

    # by-seed lines
    if 'seed' in df.columns:
        sdata = df[['seed','α','H']].dropna().copy()
        sdata['α'] = pd.to_numeric(sdata['α'], errors='coerce')
        sdata['H'] = pd.to_numeric(sdata['H'], errors='coerce')
        sdata = sdata.dropna()
        if not sdata.empty:
            plt.figure()
            for s, g in sdata.groupby('seed'):
                gg = g.sort_values('α')
                if not gg.empty:
                    plt.plot(gg['α'], gg['H'], 'o-', label=f'seed={s}')
            plt.title('H vs α (by seed)')
            plt.xlabel('α'); plt.ylabel('H'); plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOTDIR / "H_vs_alpha_by_seed.png", dpi=160)
            plt.close()

# ---------- Alerts (optional, only if we have seeds + metrics) ----------
alerts = []
if {'seed','R','K'}.issubset(df.columns):
    stab = df.groupby(['α'])[['R','K']].std().reset_index()
    bad = stab[(stab['R'] > 0.01) | (stab['K'] > 0.08)]
    for _,r in bad.iterrows():
        alerts.append(f"ALERT: high seed variance at α={r['α']}: stdR={r['R']:.3f}, stdK={r['K']:.3f}")
if alerts:
    with open(OUTDIR / "phase12_alerts.txt","w") as f: f.write("\n".join(alerts))

# ---------- Print summary JSON ----------
print(json.dumps({
    "out_csv": str(OUTCSV),
    "alpha_star_csv": str(alpha_star_path) if not alpha_star_df.empty else None,
    "report_card": str(REPORT),
    "plots_dir": str(PLOTDIR),
    "debug_agg": str(OUTDIR / "phase12_H_agg_by_alpha.csv") if ('H' in df.columns and 'α' in df.columns) else None,
    "columns_detected": sorted(df.columns.tolist())
}, indent=2))
