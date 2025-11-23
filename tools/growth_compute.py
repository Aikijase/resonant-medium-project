#!/usr/bin/env python3
"""
growth_compute.py — Solve linear growth D(a) and f(a) given a background expansion.

Modes:
  1) LCDM baseline:
       --mode lcdm --omega-m0 0.3 --sigma8 0.811 --zmax 2.0 --N 1200
  2) From table (model-agnostic):
       --mode table --table path/to/bg.csv --omega-m0 0.3 --sigma8 0.811

Outputs:
  <out-prefix>_growth_fsigma8.csv  (columns: z, a, D, f, f_sigma8)
  <out-prefix>_growth_fsigma8.png  (plot)
"""

import argparse, csv, math, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# --- Helpers -----------------------------------------------------

def read_table(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return rows, r.fieldnames

def interp_lin(x, y):
    x = np.asarray(x); y = np.asarray(y)
    def fn(q):
        return np.interp(q, x, y)
    return fn

def build_E2_from_table(path):
    rows, cols = read_table(path)
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
    order = np.argsort(A); A = A[order]; E2 = E2[order]

    if cH and not cE2:
        idx = np.argmin(np.abs(A-1.0))
        H1 = E2[idx]
        E2 = (E2/H1)**2

    # ensure E2(1)=1
    idx = np.argmin(np.abs(A-1.0))
    if E2[idx] != 0:
        E2 = E2/E2[idx]

    return interp_lin(A, E2), A.min(), A.max()

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

def solve_growth(a_eval_increasing, E2f, om0):
    """Integrate from a_ini -> 1 with an increasing a grid."""
    a_ini = max(1e-3, float(a_eval_increasing[0]))
    if not np.all(np.diff(a_eval_increasing) >= 0):
        raise ValueError("a_eval_increasing must be sorted ascending")
    if a_eval_increasing[-1] > 1.0 + 1e-12:
        raise ValueError("a grid must end at <= 1")
    # ensure last point is exactly 1.0 for clean normalization
    if abs(a_eval_increasing[-1] - 1.0) > 1e-12:
        a_eval = np.concatenate([a_eval_increasing, [1.0]])
    else:
        a_eval = a_eval_increasing

    y0 = [a_ini, 1.0]  # EdS-like IC: D~a, dD/da~1 at early times

    def rhs(a, y):
        D, dDa = y
        d2Da = -(3/a + dlnH_dla(a,E2f)/a)*dDa + 1.5*Omega_m_of_a(a,om0,E2f)/a**2*D
        return [dDa, d2Da]

    sol = solve_ivp(rhs, (a_ini, 1.0), y0, t_eval=a_eval, rtol=1e-6, atol=1e-9)
    if not sol.success:
        raise RuntimeError("Growth ODE failed: " + sol.message)

    D_raw = sol.y[0]
    D = D_raw / D_raw[-1]
    dDa = sol.y[1] / D_raw[-1]
    f = (a_eval / D) * dDa
    return a_eval, D, f

# --- Main --------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lcdm","table"], required=True)
    ap.add_argument("--omega-m0", type=float, required=True)
    ap.add_argument("--sigma8", type=float, required=True)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=1200)
    ap.add_argument("--table", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default="outputs/thrace_best/growth")
    args = ap.parse_args()

    Path(Path(args.out_prefix).parent).mkdir(parents=True, exist_ok=True)

    # Build E2(a)
    if args.mode == "lcdm":
        E2f = E2_LCDM(args.omega_m0)
    else:
        if args.table is None:
            print("ERROR: --table is required in table mode", file=sys.stderr)
            sys.exit(2)
        E2f, _, _ = build_E2_from_table(args.table)

    # We want output at z in [0, zmax] ascending, which means a descending.
    # For the ODE, we create an increasing a grid from a_min to 1.
    z_out = np.linspace(0.0, args.zmax, args.N)          # ascending
    a_out_desc = 1.0/(1.0 + z_out)                       # descending
    a_min = max(1e-3, float(a_out_desc.min()))
    a_eval_inc = np.linspace(a_min, 1.0, args.N)         # increasing for solver

    # Solve on increasing a
    a_eval_inc, D_inc, f_inc = solve_growth(a_eval_inc, E2f, args.omega_m0)

    # Interpolate results back onto the desired (descending) a grid, then reorder by z ascending
    D_desc = np.interp(a_out_desc, a_eval_inc, D_inc)
    f_desc = np.interp(a_out_desc, a_eval_inc, f_inc)
    fs8_desc = f_desc * (args.sigma8 * D_desc)

    # (z_out is already ascending and corresponds to a_out_desc)
    out_csv = f"{args.out_prefix}_growth_fsigma8.csv"
    with open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["z","a","D","f","f_sigma8"])
        for zi, ai, Di, fi, fs in zip(z_out, a_out_desc, D_desc, f_desc, fs8_desc):
            w.writerow([f"{zi:.6f}", f"{ai:.6f}", f"{Di:.6f}", f"{fi:.6f}", f"{fs:.6f}"])

    # Plot
    plt.figure()
    plt.plot(z_out, fs8_desc, label="fσ8(z)")
    plt.xlabel("z"); plt.ylabel("fσ8"); plt.title("Growth prediction")
    plt.legend(); plt.tight_layout()
    out_png = f"{args.out_prefix}_growth_fsigma8.png"
    plt.savefig(out_png, dpi=150)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()
