#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path
import numpy as np

C = 299792.458

def read_csv_rows(p):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r); cols = r.fieldnames
    return rows, cols

def sniff(cols, *cands):
    for c in cands:
        if c in cols: return c
    return None

def build_E2_from_table(path):
    rows, cols = read_csv_rows(path)
    ca = "a" if "a" in cols else None
    cz = "z" if "z" in cols else None
    cE2= "E2" if "E2" in cols else None
    cH = "H"  if "H"  in cols else None
    if not (ca or cz): raise ValueError("table must have 'a' or 'z'")
    if not (cE2 or cH): raise ValueError("table must include 'E2' or 'H'")
    A,V=[],[]
    for r in rows:
        a = float(r[ca]) if ca else 1.0/(1.0+float(r[cz]))
        v = float(r[cE2]) if cE2 else float(r[cH])
        A.append(a); V.append(v)
    A = np.asarray(A); V = np.asarray(V)
    s = np.argsort(A); A,V = A[s],V[s]
    if cH and not cE2:
        H1 = V[np.argmin(np.abs(A-1.0))]
        V = (V/H1)**2
    V = V/ V[np.argmin(np.abs(A-1.0))]
    return lambda a: np.interp(a, A, V)

def H_of_z(z, H0, E2f):
    a = 1/(1+z)
    return H0*np.sqrt(max(E2f(a),1e-300))

def DM_of_z(z, H0, E2f):
    if z<=0: return 0.0
    zs = np.linspace(0, z, max(64, int(4096*z)+32))
    Hz = np.array([H_of_z(zi, H0, E2f) for zi in zs])
    return C*np.trapz(1.0/Hz, zs)

def predict_blocks(obs_rows, cols, H0, rdrag, E2f):
    zc  = sniff(cols,"z","z_eff","zeff","zmid")
    dm_c= sniff(cols,"DM_over_rd","DM/rd","D_M_over_rdrag","D_M_over_rd","DMrd")
    dh_c= sniff(cols,"DH_over_rd","DH/rd","D_H_over_rdrag","D_H_over_rd","DHrd")
    dv_c= sniff(cols,"DV_over_rd","DV/rd","D_V_over_rdrag","D_V_over_rd","DVrd")
    dm_e= sniff(cols,"err_DM_over_rd","sigma_DM_over_rd")
    dh_e= sniff(cols,"err_DH_over_rd","sigma_DH_over_rd")
    dv_e= sniff(cols,"err_DV_over_rd","sigma_DV_over_rd")

    obs, mod, sig = [], [], []
    for r in obs_rows:
        z = r.get(zc,"")
        try: z = float(z)
        except: continue
        DM_rd = DM_of_z(z, H0, E2f)/rdrag
        DH_rd = (C/H_of_z(z, H0, E2f))/rdrag
        DV_rd = (DM_of_z(z,H0,E2f)**2 * C*z / H_of_z(z,H0,E2f))**(1/3)/rdrag

        def add(yc, ec, mval):
            if yc and ec and r.get(yc,"") not in ("",None) and r.get(ec,"") not in ("",None):
                try:
                    y=float(r[yc]); s=float(r[ec])
                except: return
                if not np.isfinite(s) or s<=0: return
                obs.append(y); mod.append(mval); sig.append(s)

        add(dm_c, dm_e, DM_rd)
        add(dh_c, dh_e, DH_rd)
        add(dv_c, dv_e, DV_rd)
    return np.array(obs), np.array(mod), np.array(sig)

def chi2_diag(obs, mod, sig): 
    r = (mod-obs)/sig
    return float(r@r), len(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--H0-grid", type=str, default="60,82,45")
    ap.add_argument("--rdrag-grid", type=str, default="130,160,45")
    args=ap.parse_args()

    H0_lo,H0_hi,H0_n = [float(x) for x in args.H0_grid.split(",")]
    rd_lo,rd_hi,rd_n = [float(x) for x in args.rdrag_grid.split(",")]
    Hs = np.linspace(H0_lo, H0_hi, int(H0_n))
    Rs = np.linspace(rd_lo, rd_hi, int(rd_n))

    E2f = build_E2_from_table(args.table)
    rows, cols = read_csv_rows(args.obs)

    best = (1e99,None,None,None)
    for H0 in Hs:
        for rd in Rs:
            obs,mod,sig = predict_blocks(rows, cols, H0, rd, E2f)
            chi2, n = chi2_diag(obs,mod,sig)
            if chi2 < best[0]:
                best = (chi2,H0,rd,n)

    chi2,H0,rd,n = best
    print(f"[fit] best χ²={chi2:.3f}  n={n}  χ²/pt={chi2/max(n,1):.3f}  H0={H0:.2f}  r_d={rd:.2f}")
    # write a tiny report
    Path("outputs/thrace_best").mkdir(parents=True, exist_ok=True)
    with open("outputs/thrace_best/bao_fit_scale.txt","w") as f:
        f.write(f"best_chi2,{chi2}\npoints,{n}\nchi2_per_point,{chi2/max(n,1)}\nH0,{H0}\nr_d,{rd}\n")

if __name__ == "__main__":
    main()
