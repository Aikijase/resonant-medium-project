#!/usr/bin/env python3
import argparse, json, pathlib as p

def main():
    ap = argparse.ArgumentParser(description="Compare IC metrics from two fit JSONs.")
    ap.add_argument("--lcdm", required=True, help="LCDM output JSON")
    ap.add_argument("--resn", required=True, help="Resonant output JSON")
    args = ap.parse_args()

    def J(x): return json.loads(p.Path(x).expanduser().read_text())
    L = J(args.lcdm); R = J(args.resn)

    def row(name, j):
        ic = j["ic"]; br = j.get("breakdown", {})
        aicc = ic.get("AICc_data", ic["AIC_data"])
        print(f"{name:28s} AIC={ic['AIC_data']:.3f}  BIC={ic['BIC_data']:.3f}  AICc={aicc:.3f}  chi2={ic['chi2_data']:.3f}  k={ic['k_params']}")
        if br:
            print(f"   -> chi2_bao={br.get('chi2_bao','-')}  chi2_sn={br.get('chi2_sn','-')}")

    row("LCDM", L)
    row("RESN", R)
    dAIC  = R["ic"]["AIC_data"] - L["ic"]["AIC_data"]
    dBIC  = R["ic"]["BIC_data"] - L["ic"]["BIC_data"]
    dAICc = R["ic"].get("AICc_data",R["ic"]["AIC_data"]) - L["ic"].get("AICc_data",L["ic"]["AIC_data"])
    print("Δ(RESN-LCDM):  ΔAIC=%.3f  ΔBIC=%.3f  ΔAICc=%.3f" % (dAIC, dBIC, dAICc))

if __name__ == "__main__":
    main()
