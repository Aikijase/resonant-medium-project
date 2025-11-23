cat > tools/sweep_gamma_profile.py <<'PY'
#!/usr/bin/env python3
import json, pathlib, subprocess as sp, numpy as np, argparse, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="Path to baseline JSON config")
    ap.add_argument("--gmin", type=float, default=1.44)
    ap.add_argument("--gmax", type=float, default=2.06)
    ap.add_argument("--gstep", type=float, default=0.02)
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--Om", type=float, default=0.3)
    ap.add_argument("--Or", type=float, default=0.0)
    ap.add_argument("--Ok", type=float, default=0.0)
    ap.add_argument("--rd", type=float, default=147.1)
    ap.add_argument("--kparams", type=int, default=5)
    ap.add_argument("--priorA", type=float, default=1.0)  # prior-A-sigma
    args = ap.parse_args()

    base = pathlib.Path(args.base_config)
    if not base.exists():
        sys.exit(f"Missing: {base}")
    C = json.loads(base.read_text())

    root = pathlib.Path.cwd()
    outdir = root / "outputs"
    outdir.mkdir(exist_ok=True)

    # Resolve paths absolute for safety
    def abs_path(k):
        p = pathlib.Path(C[k])
        return str(p if p.is_absolute() else (root / p).resolve())

    bao_csv = abs_path("bao_csv")
    bao_cov = abs_path("bao_cov")
    sn_csv  = abs_path("sn_csv")
    sn_cov  = abs_path("sn_cov")

    gammas = np.round(np.arange(args.gmin, args.gmax + 1e-9, args.gstep), 3)
    for g in gammas:
        out_prefix = (outdir / f"joint_gamma_fix_{str(g).replace('.','p')}").resolve()
        cmd = [
            "python3", str((root/"joint_guard.py").resolve()),
            "--bao-csv", bao_csv,
            "--bao-cov", bao_cov,
            "--sn-csv",  sn_csv,
            "--sn-cov",  sn_cov,
            "--out-prefix", str(out_prefix),
            "--H0", str(args.H0),
            "--Om", str(args.Om),
            "--Or", str(args.Or),
            "--Ok", str(args.Ok),
            "--rd", str(args.rd),
            "--prior-A-sigma", str(args.priorA),
            "--k-params", str(args.kparams),
            "--assume-per-rd",
            "--plus-lya",
            "--prior-gamma-mean", str(float(g)),
            "--prior-gamma-sigma", "1e-6"
        ]
        print(f"[run] gamma={g:.3f} → {out_prefix.name}.json")
        sp.check_call(cmd)

    print("[done] γ sweep complete.")
if __name__ == "__main__":
    main()
PY

chmod +x tools/sweep_gamma_profile.py
