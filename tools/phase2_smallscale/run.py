# run.py
"""
Entry point: runs
- sweep_power (figures + summary)
- dwarf_counts (derivative-based; may be NaN on some setups)
- dwarf_counts_ps_direct (σ8-normalized, stable; always returns a ratio)
- shmf_counts (calibrated counts using that ratio)
Outputs go to outputs/phase2_smallscale/ by default.
"""
import argparse, json, os, yaml
from sweep_power import run_sweep
from dwarf_counts import estimate_counts
from dwarf_counts_ps_direct import estimate_counts_ps_direct
from shmf_counts import predict_counts_from_ratio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    outdir = cfg.get("outdir", "outputs/phase2_smallscale")
    tau = float(cfg.get("tau", 0.05))
    omega = float(cfg.get("omega", 0.144))
    hmf = cfg.get("hmf", "st")
    phase_smooth = bool(cfg.get("phase_smooth", True))

    # 1) Sweep/figures
    sweep_summary = run_sweep(outdir=outdir, hmf_model=hmf,
                              tau=tau, omega=omega, phase_smooth=phase_smooth)

    # 2) Dwarf counts (derivative-based; may be NaN on some machines)
    dcfg = cfg.get("dwarf_counts", {})
    Mmin = float(dcfg.get("Mmin", 1e9))
    Mmax = float(dcfg.get("Mmax", 1e10))
    Rvir = float(dcfg.get("Rvir_kpc_h", 250.0))
    counts_deriv = estimate_counts(hmf_model=hmf, tau=tau, omega=omega,
                                   Mmin=Mmin, Mmax=Mmax, Rvir_kpc_h=Rvir,
                                   phase_smooth=phase_smooth)

    # 3) PS-direct, σ8-normalized, derivative-stable
    psd = estimate_counts_ps_direct(tau=tau, omega=omega,
                                    Mmin=Mmin, Mmax=Mmax, Rvir_kpc_h=Rvir,
                                    phase_smooth=phase_smooth,
                                    sigma8_target=cfg.get("sigma8", 0.811))
    ratio = psd.get("suppression_ratio", None)

    # 4) SHMF calibrated dwarf prediction (scale an LCDM baseline)
    scfg = cfg.get("shmf_calibrated", {})
    baseline_LCDM = float(scfg.get("baseline_LCDM", 60.0))
    Mhost = float(scfg.get("Mhost", 1e12))
    shmf_out = predict_counts_from_ratio(ratio,
                                         baseline_LCDM=baseline_LCDM,
                                         Mhost=Mhost, Mmin=Mmin, Mmax=Mmax)

    combo = {
        "sweep_summary": sweep_summary,
        "dwarf_counts_derivative": counts_deriv,
        "ps_direct": psd,
        "shmf_calibrated": shmf_out
    }
    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, "combined_results.json"), "w").write(json.dumps(combo, indent=2))
    print(json.dumps(combo, indent=2))

if __name__ == "__main__":
    main()
