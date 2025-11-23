#!/usr/bin/env python3
import os, sys, csv, json, itertools, subprocess, numpy as np, matplotlib.pyplot as plt
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "phase10_phasecouple_demo.py")
OUTDIR = os.path.join(os.path.dirname(HERE), "outputs", "phase10")
os.makedirs(OUTDIR, exist_ok=True)

def run_one(args, prefix):
    p = subprocess.run([PY, RUN] + args + ["--prefix", prefix], text=True, capture_output=True)
    if p.returncode != 0: raise RuntimeError(p.stderr.strip() or "runner error")
    return json.loads(p.stdout)

def main():
    preset="neuron"; kv=0.10; kx=0.20; eps=0.06; adapt_every=15
    steps=20000; dt=0.01; burn=300; noise=0.01; seed1=1; seed2=2
    w1 = 2.8  # baseline
    # detune grid around w1
    OMEGA2 = [round(w,2) for w in np.linspace(2.2,3.2,12)]
    KPHI   = [0.2,0.3,0.4,0.5,0.7,0.9,1.1]

    rows=[]
    for w2, Kphi in itertools.product(OMEGA2, KPHI):
        args = ["--preset", preset, "--omega2", str(w2),
                "--kv", str(kv), "--kx", str(kx), "--Kphi", str(Kphi),
                "--eps", str(eps), "--adapt_every", str(adapt_every),
                "--noise", str(noise),
                "--seed1", str(seed1), "--seed2", str(seed2),
                "--steps", str(steps), "--dt", str(dt), "--burn_in", str(burn),
                "--outdir", OUTDIR]
        prefix = f"dwmap_w2_{str(w2).replace('.','p')}_K{str(Kphi).replace('.','p')}"
        js = run_one(args, prefix); m=js["metrics"]
        rows.append(dict(
            delta_omega=round(w2 - w1,3), omega2=w2, Kphi=Kphi,
            sync_index=m["sync_index"],
            mean_abs_phase_diff=m["mean_abs_phase_diff"],
            mean_abs_xdiff=m["mean_abs_xdiff"],
            timeseries=js["outputs"]["timeseries"]))

    csv_path = os.path.join(OUTDIR, "tongue_dw_kphi.csv")
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("CSV saved:", csv_path)

    # heatmap: rows=Kphi, cols=Δω
    Ks = sorted(set(r["Kphi"] for r in rows))
    Ds = sorted(set(r["delta_omega"] for r in rows))
    Z  = np.zeros((len(Ks), len(Ds)))
    for r in rows:
        i=Ks.index(r["Kphi"]); j=Ds.index(r["delta_omega"])
        Z[i,j]=r["sync_index"]

    plt.figure(figsize=(7,5))
    plt.imshow(Z, origin="lower", aspect="auto",
               extent=[min(Ds), max(Ds), min(Ks), max(Ks)])
    plt.colorbar(label="sync_index"); plt.xlabel("Δω = ω2 − 2.8"); plt.ylabel("Kphi")
    plt.title(f"Arnold Tongue (Δω, Kphi)  @ kx={kx}, kv={kv}, eps={eps}")
    png_path = os.path.join(OUTDIR, "tongue_dw_kphi_heatmap.png")
    plt.tight_layout(); plt.savefig(png_path, dpi=180); plt.close()
    print("Heatmap saved:", png_path)

    best = max(rows, key=lambda r: r["sync_index"])
    print("Best:", dict(delta_omega=best["delta_omega"], Kphi=best["Kphi"], sync_index=best["sync_index"]))
if __name__=="__main__": main()
