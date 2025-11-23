#!/usr/bin/env python3
import os, numpy as np, csv, json
C_KMS=299792.458; H0=70.0
def DM_DH(z, Om):
    E  = np.sqrt(Om*(1+z)**3 + (1-Om))
    DH = C_KMS/(H0*E)
    # simple trapz for chi(z)
    zz = np.linspace(0,z,2000); Ez=np.sqrt(Om*(1+zz)**3+(1-Om)); inv=1/np.maximum(Ez,1e-12)
    chi = np.trapz(inv, zz)*(C_KMS/H0)
    return chi, DH
def DV(z, DM, DH): return ((DM**2)*(z*DH))**(1/3)
os.makedirs("data/desi_dr1_bao", exist_ok=True)
z = np.array([0.35,0.51,0.70,0.85,1.05], float)
Om=0.31
DM, DH = [], []
for zi in z:
    dmi, dhi = DM_DH(zi, Om); DM.append(dmi); DH.append(hhi:=dhi)
DM=np.array(DM); DH=np.array(DH); DVv=DV(z,DM,DH)
# choose a fiducial r_d (the pipeline profiles beta=1/r_d anyway)
rd=147.0
DMrd=DM/rd; DHrd=DH/rd; DVrd=DVv/rd
with open("data/desi_dr1_bao/bao_measurements.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["z","DM_over_r_d","DH_over_r_d","DV_over_r_d"])
    for i in range(len(z)): w.writerow([z[i], DMrd[i], DHrd[i], DVrd[i]])
# 2% diagonal covariance
y = np.concatenate([DMrd, DHrd, DVrd])
sig = 0.02*y
C = np.diag(sig**2)
np.savetxt("data/desi_dr1_bao/bao_covariance.csv", C, delimiter=",")
print("Wrote data/desi_dr1_bao/bao_measurements.csv and bao_covariance.csv")
