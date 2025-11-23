import argparse, csv, math
import numpy as np

def read_grid(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = np.array([[float(x) for x in row] for row in r])
    a = rows[:,0]
    D_LCDM = rows[:,1]
    # parse k columns named like D_visc_k=0.100000
    k_list = []
    Dk = []
    for j,h in enumerate(header[2:], start=2):
        if h.startswith("D_visc_k="):
            k = float(h.split("=")[1])
            k_list.append(k)
            Dk.append(rows[:,j])
    Dk = np.array(Dk)  # shape (nk, na)
    return a, D_LCDM, np.array(k_list), Dk

def nearest_idx(xarr, x):
    return int(np.argmin(np.abs(xarr - x)))

def dlnD_dlnA(D, a, i):
    # central difference in ln a
    if i==0: i=1
    if i==len(a)-1: i=len(a)-2
    N = np.log(a)
    return (np.log(D[i+1]) - np.log(D[i-1])) / (N[i+1] - N[i-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="..._growth_grid.csv from the env-run or script")
    ap.add_argument("--z", required=True, help="comma list, e.g. 0,0.3,0.6,1.0")
    ap.add_argument("--k", required=True, help="comma list in h/Mpc, e.g. 0.1,0.2,0.3")
    ap.add_argument("--out", required=True, help="output CSV")
    args = ap.parse_args()

    a, D_LCDM, k_cols, Dk = read_grid(args.grid)
    z_list = [float(z) for z in args.z.split(",")]
    k_req  = [float(k) for k in args.k.split(",")]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z","a","k_req_hMpc","k_sel_hMpc",
                    "Dvisc/Dlcdm","fvisc/flcdm","(fD)_visc/(fD)_lcdm"])
        for z in z_list:
            a_z = 1.0/(1.0+z)
            ia = nearest_idx(a, a_z)
            for k in k_req:
                ik = nearest_idx(k_cols, k)
                Dv = Dk[ik, ia]; Dl = D_LCDM[ia]
                supD = Dv / Dl
                f_v = dlnD_dlnA(Dk[ik,:], a, ia)
                f_l = dlnD_dlnA(D_LCDM, a, ia)
                ratio_f  = f_v / f_l
                ratio_fD = (f_v*Dv) / (f_l*Dl)
                w.writerow([z, a[ia], k, k_cols[ik], supD, ratio_f, ratio_fD])

if __name__ == "__main__":
    main()
