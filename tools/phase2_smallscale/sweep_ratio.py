import numpy as np, json, csv
from dwarf_counts_ps_direct import estimate_counts_ps_direct

def run(tau_vals, omega_vals, Mmin=1e9, Mmax=1e10, phase_smooth=True, sigma8=0.811):
    rows=[["tau","omega","ratio","N_LCDM","N_RES"]]
    for tau in tau_vals:
        for om in omega_vals:
            out = estimate_counts_ps_direct(tau=tau, omega=om, Mmin=Mmin, Mmax=Mmax,
                                            phase_smooth=phase_smooth, sigma8_target=sigma8)
            r = out.get("suppression_ratio", None)
            rows.append([tau, om, r, out.get("ndensity_LCDM_Mpch3"), out.get("ndensity_RES_Mpch3")])
    with open("outputs/phase2_smallscale/ratio_sweep.csv","w",newline="") as f:
        csv.writer(f).writerows(rows)
    print(json.dumps({"wrote":"outputs/phase2_smallscale/ratio_sweep.csv"}, indent=2))

if __name__=="__main__":
    tau_vals = np.linspace(0.03,0.06,7)   # edit as you like
    omega_vals = np.linspace(0.11,0.16,10)
    run(tau_vals, omega_vals)
