#!/usr/bin/env python3
import argparse, csv, json, math, os, re, shlex, subprocess, sys
from typing import List, Tuple, Optional, Dict

ALIASES = {
    "omega2": ["omega2","w2","omega_sq","omega_2","omega^2","o2","freq2","w_2"],
    "Kphi":   ["kphi","k_phi","k","kval","k_value","kphi_val","kφ","k-phi"],
    "si":     ["si","sync_index","syncidx","s_index","s","si_est","si_value","sync","synchrony","sync-index"],
}
def _norm(s: str) -> str:
    s = s.strip().lower().replace("φ","phi")
    return re.sub(r"[^a-z0-9]+","", s)

def _find_col(header: List[str], want: str, override: Optional[str]) -> Optional[str]:
    if override:
        hn = { _norm(h): h for h in header }
        if override in header: return override
        if _norm(override) in hn: return hn[_norm(override)]
    hn = { _norm(h): h for h in header }
    for alias in ALIASES[want]:
        na = _norm(alias)
        if na in hn: return hn[na]
    return None

def run_demo(omega2: float, Kphi: float, steps: int, sim_args: str) -> Optional[float]:
    cmd = [
        sys.executable, "tools/phase10_phasecouple_demo.py",
        "--preset","neuron",
        "--omega2", str(omega2),
        "--Kphi",   str(Kphi),
        "--steps",  str(steps),
        "--burn_in","300",
        "--kv","0.10","--kx","0.20","--eps","0.06","--adapt_every","15","--noise","0.01",
        "--prefix","p14_refine_tmp"
    ]
    if sim_args:
        cmd += shlex.split(sim_args)
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(p.stdout)
        return float(data["metrics"]["sync_index"])
    except Exception:
        print("[warn] Simulator JSON parse failed; stdout:\n", p.stdout)
        return None

def load_shell(path: str, colmap: Dict[str,str], status_col: Optional[str], status_accept: Optional[List[str]]) -> List[dict]:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        for key in ("omega2","Kphi","si"):
            if not colmap.get(key):
                colmap[key] = _find_col(header, key, None)
        missing = [k for k in ("omega2","Kphi","si") if not colmap[k]]
        if missing:
            print("Shell CSV header:", header)
            print("Missing:", missing)
            raise SystemExit("Shell CSV must contain omega2,Kphi,si columns (or use --col-*)")
        if status_col and status_col not in header:
            print("[warn] status column not found:", status_col)
            status_col = None

        for row in r:
            if status_col and status_accept:
                val = (row.get(status_col) or "").strip().lower()
                if val not in status_accept:
                    continue
            try:
                rows.append({
                    "omega2": float(row[colmap["omega2"]]),
                    "Kphi":  float(row[colmap["Kphi"]]),
                    "si":    float(row[colmap["si"]]),
                })
            except Exception:
                continue
    if not rows:
        raise SystemExit("No valid rows loaded from shell CSV.")
    return rows

def bisection_on_segment(a: dict, b: dict, target: float, steps: int, tol: float, max_iters: int, sim_args: str):
    w1, k1, s1 = a["omega2"], a["Kphi"], a["si"]
    w2, k2, s2 = b["omega2"], b["Kphi"], b["si"]
    if abs(s1 - target) <= tol: return w1, k1, s1, 0
    if abs(s2 - target) <= tol: return w2, k2, s2, 0
    lo_t, hi_t = 0.0, 1.0
    lo_s, hi_s = s1, s2
    evals = 0
    for _ in range(max_iters):
        mid_t = 0.5*(lo_t+hi_t)
        wm = w1 + (w2 - w1)*mid_t
        km = k1 + (k2 - k1)*mid_t
        sm = run_demo(wm, km, steps, sim_args); evals += 1
        if sm is None:
            hi_t = mid_t
            continue
        if abs(sm - target) <= tol:
            return wm, km, sm, evals
        if (sm - target)*(lo_s - target) < 0:
            hi_t, hi_s = mid_t, sm
        else:
            lo_t, lo_s = mid_t, sm
    mid_t = 0.5*(lo_t+hi_t)
    wm = w1 + (w2 - w1)*mid_t
    km = k1 + (k2 - k1)*mid_t
    sm = run_demo(wm, km, steps, sim_args); evals += 1 if sm is not None else 0
    return wm, km, (sm if sm is not None else float("nan")), evals

def bracket_adjacent(rows: List[dict], target: float) -> Optional[Tuple[int,int]]:
    for i in range(len(rows)-1):
        sa, sb = rows[i]["si"], rows[i+1]["si"]
        if math.isnan(sa) or math.isnan(sb): continue
        if (sa - target) == 0: return (i,i)
        if (sa - target)*(sb - target) < 0 or (sb - target) == 0:
            return (i, i+1)
    return None

def bracket_anypair(rows: List[dict], target: float, radius: float) -> Optional[Tuple[dict,dict]]:
    best = None
    for i in range(len(rows)):
        sa = rows[i]["si"]
        if math.isnan(sa): continue
        for j in range(i+1, len(rows)):
            sb = rows[j]["si"]
            if math.isnan(sb): continue
            if (sa - target)*(sb - target) >= 0:
                continue
            dw = rows[i]["omega2"] - rows[j]["omega2"]
            dk = rows[i]["Kphi"]  - rows[j]["Kphi"]
            dist = (dw*dw + dk*dk) ** 0.5
            if dist > radius:
                continue
            if best is None or dist < best[0]:
                best = (dist, rows[i], rows[j])
    if best is None:
        return None
    return best[1], best[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell-csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="p14_edge")
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--col-omega2", default=None)
    ap.add_argument("--col-Kphi", default=None)
    ap.add_argument("--col-si", default=None)
    ap.add_argument("--status-col", default=None)
    ap.add_argument("--status-accept", default=None)
    ap.add_argument("--nearest", action="store_true")
    ap.add_argument("--sim-args", default="")
    ap.add_argument("--any-pair", action="store_true", help="Allow any bracketing pair (not just adjacent)")
    ap.add_argument("--pair-radius", type=float, default=0.35, help="Max distance for any-pair bracketing")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    accept = [s.strip().lower() for s in (args.status_accept.split(",") if args.status_accept else [])] or None
    colmap = {"omega2": args.col_omega2, "Kphi": args.col_Kphi, "si": args.col_si}
    rows = load_shell(args.shell_csv, colmap, args.status_col, accept)

    # Try adjacent; if fail and any-pair enabled, try any pair within radius
    idx = bracket_adjacent(rows, args.target)
    mode = "adjacent"
    if idx is None and args.any_pair:
        pair = bracket_anypair(rows, args.target, args.pair_radius)
        if pair is None:
            if args.nearest:
                j = min(range(len(rows)), key=lambda i: abs(rows[i]["si"]-args.target))
                r = rows[j]
                w,k,s,evals = r["omega2"], r["Kphi"], r["si"], 0
                mode = "nearest"
            else:
                raise SystemExit("No bracketing pair found (adjacent or within radius). Try --nearest or grow shell.")
        else:
            a,b = pair
            w,k,s,evals = bisection_on_segment(a,b,args.target,args.steps,args.tol,args.max_iters,args.sim_args)
            mode = "any-pair"
    elif idx is None:
        if args.nearest:
            j = min(range(len(rows)), key=lambda i: abs(rows[i]["si"]-args.target))
            r = rows[j]
            w,k,s,evals = r["omega2"], r["Kphi"], r["si"], 0
            mode = "nearest"
        else:
            raise SystemExit("No bracketing adjacent rows found around target in polyline mode.")
    else:
        i0,i1 = idx
        a = rows[i0]
        b = rows[i1] if i1 != i0 else rows[min(i0+1, len(rows)-1)]
        w,k,s,evals = bisection_on_segment(a,b,args.target,args.steps,args.tol,args.max_iters,args.sim_args)

    out_csv = os.path.join(args.outdir, f"{args.prefix}_edge_si{args.target:.3f}.csv")
    with open(out_csv, "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["omega2","Kphi","si","mode","evals"]); wtr.writerow([w,k,s,mode,evals])

    out_png = out_csv.replace(".csv",".png")
    py = f"""
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("{args.shell_csv}")
plt.figure()
plt.scatter(df.get("{colmap['omega2'] or 'omega2'}"), df.get("{colmap['Kphi'] or 'Kphi'}"), s=10, alpha=0.6)
pt = pd.read_csv("{out_csv}")
plt.scatter(pt["omega2"], pt["Kphi"], s=80, marker='x')
plt.title("Refined edge @ si={args.target} ({mode})")
plt.xlabel("{colmap['omega2'] or 'omega2'}"); plt.ylabel("{colmap['Kphi'] or 'Kphi'}")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig("{out_png}", dpi=150)
"""
    subprocess.run([sys.executable,"-c",py])
    print("=== edge trace summary ===")
    print(f"rows_used=1  rows_considered={len(rows)}  evals≈{evals}")
    print(f"csv: {out_csv}")
    print(f"png: {out_png}")

if __name__ == "__main__":
    main()
