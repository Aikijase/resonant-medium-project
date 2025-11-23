#!/usr/bin/env python3
import argparse, math, numpy as np, pandas as pd

C_KM_S = 299792.458

def E_of_z(z, Om0): return math.sqrt(Om0*(1+z)**3 + (1-Om0))
def H_of_z(z, h, Om0): return 100.0*h*E_of_z(z, Om0)
def chi_comoving(z, h, Om0, n=2048):
    if z==0: return 0.0
    zz = np.linspace(0, z, n)
    Ez = np.sqrt(Om0*(1+zz)**3 + (1-Om0))
    return (C_KM_S/(100.0*h)) * np.trapz(1.0/Ez, zz)
def LCDM_dist(z, Om0, h):
    DM = np.array([chi_comoving(float(zi), h, Om0) for zi in z])
    DH = C_KM_S / np.array([H_of_z(float(zi), h, Om0) for zi in z])
    DV = (DM**2 * (C_KM_S*z/np.array([H_of_z(float(zi), h, Om0) for zi in z])))**(1/3)
    return DM, DH, DV

def read_wide(path):
    df = pd.read_csv(path)
    def pick(names):
        low = {c.lower().strip(): c for c in df.columns}
        for n in names:
            if n in low: return low[n]
        return None
    zc  = pick(["z","z_eff","redshift"])
    DMc = pick(["dm_over_r_d","dm_over_rd","dm_over_rdrag","dm"])
    DHc = pick(["dh_over_r_d","dh_over_rd","dh_over_rdrag","dh","hz_over_rd","hz_over_r_d"])
    DVc = pick(["dv_over_r_d","dv_over_rd","dv_over_rdrag","dv"])
    z = pd.to_numeric(df[zc], errors="coerce").to_numpy()
    DM = pd.to_numeric(df[DMc], errors="coerce").to_numpy() if DMc else None
    DH = pd.to_numeric(df[DHc], errors="coerce").to_numpy() if DHc else None
    DV = pd.to_numeric(df[DVc], errors="coerce").to_numpy() if DVc else None
    return z, DM, DH, DV

def build_pair(z, DM, DH, DV):
    y, kinds, zz = [], [], []
    for i in range(len(z)):
        if DM is not None and np.isfinite(DM[i]): y.append(float(DM[i])); kinds.append("DM"); zz.append(float(z[i]))
        if DH is not None and np.isfinite(DH[i]): y.append(float(DH[i])); kinds.append("DH"); zz.append(float(z[i]))
        if DV is not None and np.isfinite(DV[i]): y.append(float(DV[i])); kinds.append("DV"); zz.append(float(z[i]))
    return np.array(zz), kinds, np.array(y)

def build_block(z, DM, DH, DV):
    y, kinds, zz = [], [], []
    if DM is not None:
        for i in range(len(z)):
            if np.isfinite(DM[i]): y.append(float(DM[i])); kinds.append("DM"); zz.append(float(z[i]))
    if DH is not None:
        for i in range(len(z)):
            if np.isfinite(DH[i]): y.append(float(DH[i])); kinds.append("DH"); zz.append(float(z[i]))
    if DV is not None:
        for i in range(len(z)):
            if np.isfinite(DV[i]): y.append(float(DV[i])); kinds.append("DV"); zz.append(float(z[i]))
    return np.array(zz), kinds, np.array(y)

def chi2_for_stack(z, kinds, y, C, Om0, h):
    DM, DH, DV = LCDM_dist(z, Om0, h)
    m=[]; iDM=iDH=iDV=0
    for k in kinds:
        if k=="DM": m.append(DM[iDM]); iDM+=1
        elif k=="DH": m.append(DH[iDH]); iDH+=1
        elif k=="DV": m.append(DV[iDV]); iDV+=1
        else: raise RuntimeError("unexpected kind")
    m = np.array(m); y=np.array(y)
    eps = 1e-12 * float(np.median(np.diag(C)))
    Cf = C + eps*np.eye(C.shape[0])
    Ci = np.linalg.inv(Cf)
    beta = float((m @ (Ci @ y)) / max(m @ (Ci @ m), 1e-300))
    r = y - beta*m
    return float(r @ (Ci @ r)), beta

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bao-csv", default="bao_measurements.csv")
    ap.add_argument("--cov-csv", default="bao_covariance.csv")
    ap.add_argument("--Om0", type=float, default=0.3)
    ap.add_argument("--h",   type=float, default=0.7)
    args = ap.parse_args()

    z, DM, DH, DV = read_wide(args.bao_csv)
    C = np.loadtxt(args.cov_csv, delimiter=",").astype(float)

    z_p,k_p,y_p = build_pair(z,DM,DH,DV)
    z_b,k_b,y_b = build_block(z,DM,DH,DV)

    c2p,bp = chi2_for_stack(z_p,k_p,y_p,C,args.Om0,args.h)
    c2b,bb = chi2_for_stack(z_b,k_b,y_b,C,args.Om0,args.h)

    dof_p = len(y_p)-1; dof_b=len(y_b)-1
    print(f"PAIR : χ²={c2p:.3f}  dof≈{dof_p}  χ²/dof={c2p/dof_p:.3f}  β={bp:.6g}")
    print(f"BLOCK: χ²={c2b:.3f}  dof≈{dof_b}  χ²/dof={c2b/dof_b:.3f}  β={bb:.6g}")
    print("=> preferred:", "PAIR" if c2p< c2b else "BLOCK")
if __name__ == "__main__":
    main()
