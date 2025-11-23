#!/usr/bin/env python3
"""
Phase-27: Recycling – WWI Scoring (config-driven)

Reads thresholds/scales from thresholds.json and scores Phase-23 predictions.
Outputs:
  - outputs/phase27/bench_wwi_cfg.json
  - outputs/phase27/bench_wwi_cfg.txt
  - outputs/phase27/bench_wwi_cfg_bar.png
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def load_preds(pred_csv):
    txt = Path(pred_csv).with_name("predictions.txt")
    dneff=al=None
    if txt.exists():
        for line in open(txt):
            line=line.strip()
            if line.startswith("ΔNeff_proxy"): dneff=float(line.split("=")[1])
            if line.startswith("A_L_proxy"):    al=float(line.split("=")[1])
    df=pd.read_csv(pred_csv)
    return {"z":df["z"].tolist(), "fs8":df["fs8_proxy"].tolist(), "dneff":dneff, "al":al}

def score(preds, cfg):
    fs8_devs=[abs(1.0-x) for x in preds["fs8"]]
    mdev=float(np.mean(fs8_devs)) if fs8_devs else 1.0
    fs8_scale=max(1e-6,float(cfg.get("fs8_dev_scale",0.10)))
    al_scale =max(1e-6,float(cfg.get("al_var_scale",0.05)))
    dn_scale =max(1e-6,float(cfg.get("dneff_scale",0.20)))
    al = preds["al"] if preds["al"] is not None else al_scale
    dneff = preds["dneff"] if preds["dneff"] is not None else dn_scale

    fs8_score = 100.0*max(0.0,1.0 - (mdev/fs8_scale))
    al_score  = 100.0*max(0.0,1.0 - (al/al_scale))
    dn_score  = 100.0*max(0.0,1.0 - (dneff/dn_scale))
    wwi = 0.50*fs8_score + 0.25*al_score + 0.25*dn_score
    return {"fs8_score":fs8_score,"al_score":al_score,"dneff_score":dn_score,"WWI":min(100.0,float(wwi))}

def main():
    ap=argparse.ArgumentParser(description="Phase-27: WWI (config-driven)")
    ap.add_argument("--pred-csv", default="outputs/phase23/predictions.csv")
    ap.add_argument("--fit-json", default="outputs/phase22/recycling_fit.json")
    ap.add_argument("--cfg", default="thresholds.json")
    ap.add_argument("--outdir", default="outputs/phase27")
    args=ap.parse_args()

    print("=== Phase-27: Recycling – WWI (config-driven) ===")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.pred_csv):
        print("[ERROR] Missing predictions CSV. Run Phase-23 first."); print("RESULT: FAIL"); sys.exit(2)
    preds=load_preds(args.pred_csv)

    cfg=json.load(open(args.cfg)) if os.path.exists(args.cfg) else {}
    pass_thres=float(cfg.get("pass_threshold",60.0))

    best={}
    if os.path.exists(args.fit_json):
        try: best=(json.load(open(args.fit_json)).get("best") or {})
        except: best={}

    S=score(preds,cfg)
    status="PASS" if S["WWI"]>=pass_thres else "FAIL"

    J={"ok":True,"status":status,"threshold":pass_thres,"scores":S,"preds":preds,"best_params":best}
    jpath=Path(args.outdir)/"bench_wwi_cfg.json"; open(jpath,"w").write(json.dumps(J,indent=2))
    tpath=Path(args.outdir)/"bench_wwi_cfg.txt"
    with open(tpath,"w") as f:
        if best: f.write(f"Best (22): k={best.get('k')} alpha={best.get('alpha')} eps={best.get('epsilon')}\n")
        f.write(f"fs8_score={S['fs8_score']:.2f}\nA_L_score={S['al_score']:.2f}\nΔNeff_score={S['dneff_score']:.2f}\nWWI={S['WWI']:.2f}\n")
        f.write(f"Threshold={pass_thres:.1f}\nRESULT: {status}\n")

    # Bar plot
    ppath=Path(args.outdir)/"bench_wwi_cfg_bar.png"
    labels=["fs8","A_L","ΔNeff","WWI"]; ours=[S["fs8_score"],S["al_score"],S["dneff_score"],S["WWI"]]; base=[100,100,100,100]
    x=np.arange(len(labels)); w=0.35
    plt.figure(figsize=(8,5))
    plt.bar(x-w/2, base, width=w, label="ΛCDM (ref)")
    plt.bar(x+w/2, ours, width=w, label="Recycling")
    plt.xticks(x,labels); plt.ylabel("Score"); plt.ylim(0,110); plt.title("Phase-27: WWI (config-driven)")
    plt.legend(); plt.tight_layout(); plt.savefig(ppath,dpi=140)

    print(f"Wrote {jpath}"); print(f"Wrote {tpath}"); print(f"Wrote {ppath}")
    print(f"\nRESULT: {status}")

if __name__=="__main__":
    main()
