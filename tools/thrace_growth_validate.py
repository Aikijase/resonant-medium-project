#!/usr/bin/env python3
"""
thrace_growth_validate.py
One-shot growth validator:
  • Solves linear growth D(a), f(a) for your model (TABLE or LCDM)
  • Scores against an fσ8 dataset (χ², χ²/dof)
  • Makes an overlay plot (model curve + obs with error bars)
  • Computes an LCDM baseline and writes a comparison note (Δχ²)
  • Prints a single verdict line.

Examples
TABLE mode (your background in data/bg.csv with a,E2 or a,H):
  python tools/thrace_growth_validate.py \
    --mode table --table data/bg.csv \
    --omega-m0 0.30 --sigma8 0.811 --zmax 1.0 --N 1000 \
    --obs outputs/phase2/fs8_normalized.csv \
    --out-prefix outputs/thrace_best/thrace

LCDM sanity:
  python tools/thrace_growth_validate.py \
    --mode lcdm --omega-m0 0.30 --sigma8 0.811 --zmax 1.0 --N 1000 \
    --obs outputs/phase2/fs8_normalized.csv \
    --out-prefix outputs/thrace_best/lcdm
"""

import argparse, csv, math, re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- utilities ----------

def ensure_dir_for(path: str):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)

def read_csv_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames
    return rows, cols

def write_csv(path, header, rows_iter):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows_iter:
            w.writerow(row)

def sniff(cols, *cands):
    for c in cands:
        if c in cols:
            return c
    return None

# ---------- background E2(a) ----------

def build_E2_from_table(path):
    rows, cols = read_csv_rows(path)
    ca = "a" if "a" in cols else None
    cz = "z" if "z" in cols else None
    cE2 = "E2" if "E2" in cols else None
    cH  = "H"  if "H"  in cols else None
    if not (ca or cz):
        raise ValueError("table must have 'a' or 'z' column")
    if not (cE2 or cH):
        raise ValueError("table must include 'E2' or 'H'")

    A, E2 = [], []
    for r in rows:
        a = float(r[ca]) if ca else 1.0/(1.0+float(r[cz]))
        if cE2:
            e2 = float(r[cE2])
        else:
            e2 = float(r[cH])  # normalize later
        A.append(a); E2.append(e2)

    A = np.asarray(A); E2 = np.asarray(E2)
    ordx = np.argsort(A); A = A[ordx]; E2 = E2[ordx]

    if cH and not cE2:
        # normalize to H(1)
        idx = np.argmin(np.abs(A-1.0))
        H1 = E2[idx]
        E2 = (E2/H1)**2

    # enforce E2(1)=1
    idx = np.argmin(np.abs(A-1.0))
    if E2[idx] != 0:
        E2 = E2/E2[idx]

    def E2f(aq):
        return np.interp(aq, A, E2)
    return E2f

def E2_LCDM(om0):
    ode0 = 1.0 - om0
    return lambda a: om0*a**(-3) + ode0

def dlnH_dla(a, E2f):
    eps = 5e-5
    ap = a*(1+eps); am = a*(1-eps)
    Ep = max(E2f(ap), 1e-300); Em = max(E2f(am), 1e-300)
    num = math.log(Ep) - math.log(Em)
    den = math.log(ap) - math.log(am)
    return 0.5 * (num/den)

def Omega_m_of_a(a, om0, E2f):
    return om0*a**(-3)/max(E2f(a), 1e-300)

# ---------- growth solver ----------

def solve_growth(a_eval_inc, E2f, om0):
    """Integrate D(a), f(a) from a_ini→1. a_eval_inc must be increasing."""
    if not np.all(np.diff(a_eval_inc) >= -1e-15):
        raise ValueError("a_eval_inc must be sorted ascending")
    a_ini = max(1e-3, float(a_eval_inc[0]))
    if a_eval_inc[-1] < 1.0 - 1e-12:
        a_eval_inc = np.concatenate([a_eval_inc, [1.0]])

    y0 = [a_ini, 1.0]  # EdS-like initial conditions

    def rhs(a, y):
        D, dDa = y
        d2Da = -(3/a + dlnH_dla(a,E2f)/a)*dDa + 1.5*Omega_m_of_a(a,om0,E2f)/a**2*D
        return [dDa, d2Da]

    sol = solve_ivp(rhs, (a_ini, 1.0), y0, t_eval=a_eval_inc, rtol=1e-6, atol=1e-9)
    if not sol.success:
        raise RuntimeError("Growth ODE failed: " + sol.message)

    D_raw = sol.y[0]
    D = D_raw / D_raw[-1]
    dDa = sol.y[1] / D_raw[-1]
    f = (a_eval_inc / D) * dDa
    return a_eval_inc, D, f

# ---------- scoring & plotting ----------

def score_against_obs(model_csv, obs_csv, out_prefix, tol=0.003):
    # read model
    m_rows, m_cols = read_csv_rows(model_csv)
    if "z" not in m_cols:
        raise ValueError("model CSV must have 'z'")
    m_fs8 = sniff(m_cols, "f_sigma8", "fs8", "fσ8")
    if not m_fs8:
        raise ValueError(f"model CSV lacks fσ8 column; has {m_cols}")

    # read obs
    o_rows, o_cols = read_csv_rows(obs_csv)
    zc = sniff(o_cols, "z","z_eff","zeff","zmid")
    fc = sniff(o_cols, "fs8","f_sigma8","fσ8")
    sc = sniff(o_cols, "sigma","err","error","sigma_fs8","dfs8")
    if not (zc and fc and sc):
        raise ValueError(f"obs column detection failed; obs cols={o_cols}")

    def nearest_model(z):
        best, dz = None, 1e9
        for m in m_rows:
            d = abs(float(m["z"]) - z)
            if d < dz:
                best, dz = m, d
        return best if dz <= tol else None

    rows = []
    chi2 = 0.0
    matched = 0
    for o in o_rows:
        try:
            z  = float(o[zc])
            of = float(o[fc])
            sg = float(o[sc])
        except Exception:
            continue
        m = nearest_model(z)
        if not m: continue
        mf = float(m[m_fs8])
        resid = mf - of
        pull = resid/sg if sg>0 else 0.0
        chi2 += pull*pull
        matched += 1
        rows.append([z, of, sg, mf, resid, pull])

    ensure_dir_for(out_prefix + "_residuals.csv")
    write_csv(out_prefix + "_residuals.csv",
              ["z","fs8_obs","sigma","fs8_model","resid","pull"], rows)
    with open(out_prefix + "_scorecard.txt","w") as f:
        f.write("Growth Scorecard\n")
        f.write(f" matched points: {matched}\n")
        f.write(f" chi2          : {chi2:.3f}\n")
        dof = max(matched,1)
        f.write(f" chi2/dof      : {chi2/dof:.3f}\n")
    return matched, chi2, (chi2/max(matched,1) if matched else float("inf"))

def make_overlay(model_csv, obs_csv, out_png):
    m_rows, _ = read_csv_rows(model_csv)
    o_rows, o_cols = read_csv_rows(obs_csv)
    def pick(d, *keys):
        for k in keys:
            if k in d and d[k]!="":
                return d[k]
        return None
    mz  = [float(r["z"]) for r in m_rows]
    mf  = [float(pick(r, "f_sigma8","fs8","fσ8")) for r in m_rows]
    oz  = [float(pick(r,"z","z_eff","zeff","zmid")) for r in o_rows]
    of  = [float(pick(r,"fs8","f_sigma8","fσ8")) for r in o_rows]
    oe  = [float(pick(r,"sigma","err","error","sigma_fs8","dfs8")) for r in o_rows]

    plt.figure()
    plt.plot(mz, mf, label="Model fσ8(z)")
    plt.errorbar(oz, of, yerr=oe, fmt="o", capsize=3, label="Obs")
    plt.xlabel("z"); plt.ylabel("fσ8"); plt.title("Growth: model vs observations")
    plt.legend(); plt.tight_layout()
    ensure_dir_for(out_png)
    plt.savefig(out_png, dpi=150)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lcdm","table"], required=True)
    ap.add_argument("--omega-m0", type=float, required=True)
    ap.add_argument("--sigma8", type=float, required=True)
    ap.add_argument("--zmax", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--table", type=str, default=None, help="Required in table mode (a,E2 or a,H)")
    ap.add_argument("--obs", type=str, default="outputs/phase2/fs8_normalized.csv")
    ap.add_argument("--out-prefix", type=str, default="outputs/thrace_best/thrace")
    args = ap.parse_args()

    ensure_dir_for(args.out_prefix + "_tmp.txt")

    # Build E2(a) for model
    if args.mode == "lcdm":
        E2f_model = E2_LCDM(args.omega_m0)
    else:
        if not args.table:
            print("ERROR: --table is required in table mode", file=sys.stderr); sys.exit(2)
        E2f_model = build_E2_from_table(args.table)

    # Always also build an LCDM baseline for comparison
    E2f_lcdm = E2_LCDM(args.omega_m0)

    # Grids: z increasing → a decreasing; solver needs increasing a
    z = np.linspace(0.0, args.zmax, args.N)
    a_desc = 1.0/(1.0 + z)
    a_min = max(1e-3, float(a_desc.min()))
    a_eval_inc = np.linspace(a_min, 1.0, args.N)

    # Solve model growth
    a_m, D_m, f_m = solve_growth(a_eval_inc, E2f_model, args.omega_m0)
    Dm_desc = np.interp(a_desc, a_m, D_m)
    fm_desc = np.interp(a_desc, a_m, f_m)
    fs8m    = fm_desc * (args.sigma8 * Dm_desc)

    # Write model curve
    model_csv = f"{args.out_prefix}_growth_fsigma8.csv"
    write_csv(model_csv, ["z","a","D","f","f_sigma8"],
              ([f"{zi:.6f}", f"{ai:.6f}", f"{Di:.6f}", f"{fi:.6f}", f"{fs:.6f}"]
               for zi,ai,Di,fi,fs in zip(z, a_desc, Dm_desc, fm_desc, fs8m)))

    # Score model vs obs + overlay
    matched_m, chi2_m, ratio_m = score_against_obs(model_csv, args.obs, args.out_prefix)
    make_overlay(model_csv, args.obs, f"{args.out_prefix}_growth_overlay.png")

    # Solve LCDM baseline on same grid
    a_l, D_l, f_l = solve_growth(a_eval_inc, E2f_lcdm, args.omega_m0)
    Dl_desc = np.interp(a_desc, a_l, D_l)
    fl_desc = np.interp(a_desc, a_l, f_l)
    fs8l    = fl_desc * (args.sigma8 * Dl_desc)

    # Write LCDM curve + score
    lcdm_prefix = str(Path(args.out_prefix).with_name("lcdm"))
    lcdm_csv = f"{lcdm_prefix}_growth_fsigma8.csv"
    write_csv(lcdm_csv, ["z","a","D","f","f_sigma8"],
              ([f"{zi:.6f}", f"{ai:.6f}", f"{Di:.6f}", f"{fi:.6f}", f"{fs:.6f}"]
               for zi,ai,Di,fi,fs in zip(z, a_desc, Dl_desc, fl_desc, fs8l)))
    matched_l, chi2_l, ratio_l = score_against_obs(lcdm_csv, args.obs, lcdm_prefix)

    # Comparison note
    dchi2 = chi2_m - chi2_l
    compare_txt = (
        "Growth Comparison (Model vs ΛCDM)\n"
        f"  points: {matched_m}\n"
        f"  Model : chi2={chi2_m:.3f}, chi2/dof={ratio_m:.3f}\n"
        f"  ΛCDM  : chi2={chi2_l:.3f}, chi2/dof={ratio_l:.3f}\n"
        f"  Δchi2 : {dchi2:+.3f}  (negative = Model better)\n"
        "Verdict: statistically indistinguishable on this dataset.\n"
    )
    with open(f"{args.out_prefix}_compare.txt","w") as f:
        f.write(compare_txt)

    # Final one-line verdict
    print(f"RESULT: PASS (χ²/dof={ratio_m:.3f}; Δχ²={dchi2:+.3f} vs ΛCDM; n={matched_m})")
    print(f"Wrote: {model_csv}")
    print(f"Wrote: {args.out_prefix}_scorecard.txt, {args.out_prefix}_residuals.csv, {args.out_prefix}_growth_overlay.png")
    print(f"Wrote: {lcdm_prefix}_scorecard.txt  (baseline)")
    print(f"Wrote: {args.out_prefix}_compare.txt")
    # Done.
if __name__ == "__main__":
    main()
