import argparse, json, pathlib as p
ap = argparse.ArgumentParser()
ap.add_argument("--lcdm", required=True)
ap.add_argument("--resn", required=True)
args = ap.parse_args()
def J(x): return json.loads(p.Path(x).expanduser().read_text())
L, R = J(args.lcdm), J(args.resn)
def row(n,j):
    ic = j["ic"]; aicc = ic.get("AICc_data", ic["AIC_data"])
    print(f"{n:36s}  AIC={ic['AIC_data']:.3f}  BIC={ic['BIC_data']:.3f}  AICc={aicc:.3f}  chi2={ic['chi2_data']:.3f}  k={ic['k_params']}")
row("LCDM", L); row("Resonant", R)
dAIC  = R["ic"]["AIC_data"]  - L["ic"]["AIC_data"]
dBIC  = R["ic"]["BIC_data"]  - L["ic"]["BIC_data"]
dAICc = R["ic"].get("AICc_data",R["ic"]["AIC_data"]) - L["ic"].get("AICc_data",L["ic"]["AIC_data"])
print(f"Δ(Res-LCDM):  ΔAIC={dAIC:.3f}  ΔBIC={dBIC:.3f}  ΔAICc={dAICc:.3f}")
