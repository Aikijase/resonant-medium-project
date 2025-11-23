#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re, sys, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent

# ---------- regex patterns ----------
_RX_CHI2 = [
    re.compile(r"(?:^|\s)(?:chi2|chisq|chi[_ -]?square|cost|loss|rss|sse)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
    re.compile(r"(?:^|\s)(?:-?\s*2\s*(?:ln\s*L|logL)|neg2logL|minus2logL|minus2loglike)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
    re.compile(r"(?:^|\s)(?:nll|neg(?:ative)?\s*log(?:like|likelihood))\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
    re.compile(r"(?:^|\s)(?:loglike|log[_ ]likelihood|lnL|logL)\s*[:=]\s*([0-9eE+\-\.]+)", re.I),
]
_RX_AIC  = [re.compile(r"(?:^|\s)AIC\s*[:=]\s*([0-9eE+\-\.]+)", re.I)]
_RX_BIC  = [re.compile(r"(?:^|\s)BIC\s*[:=]\s*([0-9eE+\-\.]+)", re.I)]
_RX_dAIC = [re.compile(r"(?:ΔAIC|Delta\s*AIC|dAIC|delta[_\- ]?aic)\s*[:=]\s*([0-9eE+\-\.]+)", re.I)]
_RX_dBIC = [re.compile(r"(?:ΔBIC|Delta\s*BIC|dBIC|delta[_\- ]?bic)\s*[:=]\s*([0-9eE+\-\.]+)", re.I)]

def _first_match_float(txt: str, patterns: List[re.Pattern]) -> Optional[float]:
    if not txt: return None
    for rx in patterns:
        m = rx.search(txt)
        if m:
            try: return float(m.group(1))
            except Exception: pass
    return None

def _parse_triplet_from_text(txt: str) -> Dict[str, Optional[float]]:
    return {
        "chi2": _first_match_float(txt, _RX_CHI2),
        "AIC":  _first_match_float(txt, _RX_AIC),
        "BIC":  _first_match_float(txt, _RX_BIC),
        "dAIC": _first_match_float(txt, _RX_dAIC),
        "dBIC": _first_match_float(txt, _RX_dBIC),
    }

def _parse_chi2_from_json_obj(d: Dict[str, Any]) -> Optional[float]:
    for key in ["chi2","chisq","chi_sq","chi2_total","minus2logL","neg2logL","nll","loglike","lnL","logL"]:
        if key in d and d[key] is not None:
            try: v = float(d[key])
            except Exception: continue
            if key == "nll": return 2.0*v
            if key in ["loglike","lnL","logL"]: return -2.0*v
            return v
    return None

def _parse_json_metrics(p: Path) -> Dict[str, Optional[float]]:
    try: d = json.loads(p.read_text())
    except Exception: return {"chi2":None,"AIC":None,"BIC":None,"dAIC":None,"dBIC":None}
    out = _parse_triplet_from_text(json.dumps(d))
    chi2_struct = _parse_chi2_from_json_obj(d)
    if chi2_struct is not None: out["chi2"] = chi2_struct
    return out

def _parse_csv_metrics(p: Path) -> Dict[str, Optional[float]]:
    try:
        with p.open(newline="") as f: rows = list(csv.DictReader(f))
    except Exception: return {"chi2":None,"AIC":None,"BIC":None,"dAIC":None,"dBIC":None}
    chi2 = None
    if rows:
        header = [h.lower() for h in rows[0].keys()]
        for i,h in enumerate(header):
            if "chi" in h and "2" in h:
                try: chi2 = float(list(rows[-1].values())[i])
                except Exception: pass
        if chi2 is None:
            for row in rows[:5]:
                for k,v in row.items():
                    lk = (k or "").lower()
                    if "chi" in lk and "2" in lk:
                        try: chi2 = float(v); break
                        except Exception: pass
                if chi2 is not None: break
    t = _parse_triplet_from_text(p.read_text(errors="ignore"))
    return {"chi2": chi2 if chi2 is not None else t["chi2"], "AIC": t["AIC"], "BIC": t["BIC"], "dAIC": t["dAIC"], "dBIC": t["dBIC"]}

# ---- explicit SN+BAO block parser (LCDM/RESN + deltas) ----
_RX_LCDM = re.compile(r"^LCDM\s*:\s*(\{.*\})\s*$", re.M)
_RX_RESN = re.compile(r"^RESN\s*:\s*(\{.*\})\s*$", re.M)
_RX_DCHI = re.compile(r"^Δχ².*?:\s*([0-9eE+\-\.]+)", re.M)

def _parse_py_dict_to_jsonable(s: str) -> Dict[str, Any]:
    j = s.replace("'", '"')
    j = re.sub(r'\bNone\b', 'null', j)
    j = re.sub(r'\bTrue\b', 'true', j)
    j = re.sub(r'\bFalse\b', 'false', j)
    return json.loads(j)

def _extract_models_from_text(txt: str) -> Optional[Dict[str, Any]]:
    m_l = _RX_LCDM.search(txt); m_r = _RX_RESN.search(txt)
    if not (m_l and m_r): return None
    try:
        lcdm = _parse_py_dict_to_jsonable(m_l.group(1))
        resn = _parse_py_dict_to_jsonable(m_r.group(1))
    except Exception:
        return None
    deltas = {
        "dChi2_LCDM_minus_RESN": _first_match_float(txt, [_RX_DCHI]),
        "dAIC_LCDM_minus_RESN":  _first_match_float(txt, _RX_dAIC),
        "dBIC_LCDM_minus_RESN":  _first_match_float(txt, _RX_dBIC),
    }
    return {"lcdm": lcdm, "resn": resn, "deltas": deltas}

# ---- neighbor harvesting ----
def _harvest_neighbors(fit_path: Path) -> Dict[str, Any]:
    """Search outputs/runs/*.log first, then neighbors. Skip our own std summaries."""
    out: Dict[str, Any] = {"chi2":None,"AIC":None,"BIC":None,"dAIC":None,"dBIC":None}
    details: Dict[str, Any] = {}

    deny = re.compile(r"(?:\.std_summary\.json$|\.png$|\.pdf$|\.npy$|\.npz$|\.lock$)", re.I)
    candidates: List[Path] = []

    runs_dir = PROJECT_ROOT / "outputs" / "runs"
    if runs_dir.exists():
        candidates += sorted(runs_dir.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    parent = fit_path.parent
    candidates += sorted(parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    for p in candidates:
        if not p.is_file() or deny.search(p.name): continue
        if p.suffix.lower() == ".json":
            m = _parse_json_metrics(p); txt = None
        elif p.suffix.lower() == ".csv":
            m = _parse_csv_metrics(p); txt = None
        else:
            txt = p.read_text(errors="ignore")
            m = _parse_triplet_from_text(txt)

        for k,v in m.items():
            if out.get(k) is None and v is not None:
                out[k] = v

        if txt:
            models = _extract_models_from_text(txt)
            if models and not details:
                details = models

        if out["chi2"] is not None and (out["AIC"] is not None or out["BIC"] is not None or out["dAIC"] is not None or out["dBIC"] is not None) and details:
            break

    if details:
        out["models"] = details
    return out

# ---- fitter run + fit.json locating ----
def _run_make_fit_json(python_bin: str, fitter_script: Path, config_path: Path, tag: Optional[str], extra: Optional[List[str]]):
    cmd = [python_bin, str(fitter_script), str(config_path)]
    if tag: cmd += ["--tag", tag]
    if extra: cmd += extra
    print("[thrace-fit] Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

def _find_fit_json(search_dir: Path, tag: Optional[str]) -> Optional[Path]:
    if tag:
        for p in [search_dir/f"{tag}.fit.json", PROJECT_ROOT/"outputs"/f"{tag}.fit.json", PROJECT_ROOT/f"{tag}.fit.json"]:
            if p.exists(): return p
    for base in [search_dir, PROJECT_ROOT/"outputs"]:
        if base.is_dir():
            cand = sorted(base.rglob("*.fit.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand: return cand[0]
    return None

def _summarize_primary(obj: Dict[str,Any]) -> Dict[str,Any]:
    def first(d:Dict[str,Any], keys:List[str]):
        for k in keys:
            if k in d: return d[k]
        return None
    return {
        "model": first(obj, ["model","label","name"]),
        "chi2":  first(obj, ["chi2","chi_sq","chi2_total","chisq"]),
        "AIC":   first(obj, ["AIC","aic"]),
        "BIC":   first(obj, ["BIC","bic"]),
        "n":     first(obj, ["n","n_points","dof","ndof"]),
        "source":first(obj, ["source"]),
        "dAIC":  first(obj, ["dAIC","DeltaAIC","Delta_AIC","ΔAIC","delta_aic"]),
        "dBIC":  first(obj, ["dBIC","DeltaBIC","Delta_BIC","ΔBIC","delta_bic"]),
    }

def _std_summary_path_for(fit_path: Path) -> Path:
    # Write summaries away from outputs/* so the fitter won't re-read them
    base = fit_path.name[:-len(".fit.json")] if fit_path.name.endswith(".fit.json") else fit_path.stem
    outdir = PROJECT_ROOT / "outputs" / "thrace_summaries"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{base}.std_summary.json"

def main():
    ap = argparse.ArgumentParser(description="Wrap make_fit_json.py and surface χ²/AIC/BIC (plus LCDM/RESN details).")
    ap.add_argument("--config", required=True, help="Positional JSON config for your fitter")
    ap.add_argument("--tag", default=None, help="Tag passed to the fitter; helps locate <tag>.fit.json")
    ap.add_argument("--python", default=sys.executable, help="Python to run the fitter")
    ap.add_argument("--fitter", default=str(Path("tools")/"make_fit_json.py"))
    ap.add_argument("--search-dir", default="outputs", help="Directory to search for *.fit.json")
    ap.add_argument("--std-out", default=None, help="Optional path to write standardized summary JSON")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args to pass to the fitter after --extra ...")
    args = ap.parse_args()

    config_path = PROJECT_ROOT / args.config
    fitter_path = PROJECT_ROOT / args.fitter
    search_dir  = PROJECT_ROOT / args.search_dir

    if not fitter_path.exists(): sys.exit(f"[thrace-fit] ERROR: fitter not found at {fitter_path}")
    if not config_path.exists(): sys.exit(f"[thrace-fit] ERROR: config not found at {config_path}")

    proc = _run_make_fit_json(args.python, fitter_path, config_path, args.tag, args.extra)
    if proc.stdout.strip(): print("[fitter stdout]"); print(proc.stdout)
    if proc.stderr.strip(): print("[fitter stderr]", file=sys.stderr); print(proc.stderr, file=sys.stderr)

    fit_path = _find_fit_json(search_dir=search_dir, tag=args.tag)
    if not fit_path: sys.exit(f"[thrace-fit] ERROR: Could not locate any *.fit.json under {search_dir} (tag={args.tag!r}).")

    try: fit_obj = json.loads(fit_path.read_text())
    except Exception as e: sys.exit(f"[thrace-fit] ERROR: Failed to load fit JSON at {fit_path}: {e}")

    summary = _summarize_primary(fit_obj)

    # Fallbacks: harvest logs/CSVs/JSONs nearby (prefers outputs/runs/*.log)
    need_more = any(summary.get(k) is None for k in ["chi2","AIC","BIC","dAIC","dBIC"])
    extended = _harvest_neighbors(fit_path) if need_more else {}
    for k,v in extended.items():
        if k == "models":
            summary["models"] = v
        elif summary.get(k) is None and v is not None:
            summary[k] = v

    # Canonical metric selection (prefer RESN; fallback to LCDM)
    models = summary.get("models", {})
    resn = (models or {}).get("resn", {})
    lcdm = (models or {}).get("lcdm", {})
    if resn.get("chi2") is not None:
        summary["chi2"] = float(resn["chi2"])
        summary.setdefault("AIC", resn.get("AIC"))
        summary.setdefault("BIC", resn.get("BIC"))
    elif lcdm.get("chi2") is not None:
        summary["chi2"] = float(lcdm["chi2"])
        summary.setdefault("AIC", lcdm.get("AIC"))
        summary.setdefault("BIC", lcdm.get("BIC"))

    # Expose deltas at top-level if present
    deltas = (models or {}).get("deltas", {})
    for k_src, k_dst in [("dChi2_LCDM_minus_RESN","dChi2"), ("dAIC_LCDM_minus_RESN","dAIC"), ("dBIC_LCDM_minus_RESN","dBIC")]:
        if k_src in deltas and summary.get(k_dst) is None:
            summary[k_dst] = deltas[k_src]

    std_path = Path(args.std_out) if args.std_out else _std_summary_path_for(fit_path)
    std_path.write_text(json.dumps(summary, indent=2))

    print("[thrace-fit] Fit JSON:", fit_path)
    print("[thrace-fit] Summary JSON:", std_path)
    print("[thrace-fit] Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
