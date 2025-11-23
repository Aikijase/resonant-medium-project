#!/usr/bin/env python3
"""
Phase 10 sweep runner (drop-in wrapper, no patching).
Runs parameter sweeps using tools/phase10_resonant_operator.py and writes a CSV summary.
"""
import csv, json, subprocess, sys, os, itertools

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "phase10_resonant_operator.py")
OUTDIR = os.path.join(os.path.dirname(HERE), "outputs", "phase10")
os.makedirs(OUTDIR, exist_ok=True)

def run_one(prefix, args):
    cmd = [PY, RUN] + args + ["--prefix", prefix, "--outdir", OUTDIR]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "unknown error")
    return json.loads(p.stdout)

def main():
    # edit these to taste
    preset = "neuron"
    steps, dt = 4000, 0.01
    burn = 300

    # Example sweeps (uncomment what you need)
    grid = []
    # 1) Noise sweep
    for noise in [0.00, 0.01, 0.02, 0.05, 0.10]:
        grid.append(dict(noise=noise))
    # 2) Softness sweep (example)
    # for s in [0.2,0.35,0.5,0.65,0.8,1.0]:
    #     grid.append(dict(softness=s))
    # 3) Omega sweep (example)
    # for w in [1.6,2.0,2.4,2.8,3.2]:
    #     grid.append(dict(omega=w))

    rows = []
    for i, params in enumerate(grid, 1):
        args = ["--preset", preset, "--steps", str(steps), "--dt", str(dt), "--burn_in", str(burn)]
        for k,v in params.items():
            args += [f"--{k}", str(v)]
        prefix = "sweep_" + "_".join(f"{k}{v}" for k,v in params.items()).replace(".","p")
        js = run_one(prefix, args)
        m = js["metrics"]
        # simple SNR proxy: peak amplitude / RMS
        snr = (m["amp_dom"] / (m["rms"] + 1e-12)) if m["rms"]>0 else 0.0
        rows.append(dict(
            preset=js["preset"],
            prefix=prefix,
            steps=js["steps"], dt=js["dt"],
            noise=params.get("noise",""),
            memory=js["params"]["memory"],
            softness=js["params"]["softness"],
            omega=js["params"]["omega"],
            gamma=js["params"]["gamma"],
            gain=js["params"]["gain"],
            rms=m["rms"], mean=m["mean"],
            f_dom=m["f_dom"], amp_dom=m["amp_dom"],
            snr=snr, stability=m["stability"],
            timeseries=js["outputs"]["timeseries"],
            phase=js["outputs"]["phase"],
            spectrum=js["outputs"]["spectrum"],
        ))

    csv_path = os.path.join(OUTDIR, "phase10_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("CSV saved:", csv_path)

if __name__ == "__main__":
    main()
