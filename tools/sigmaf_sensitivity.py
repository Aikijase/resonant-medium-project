#!/usr/bin/env python3
import json, pathlib, subprocess as sp, argparse

def bump_sigma_and_run(base_cfg: pathlib.Path, sigma_f: float, out_json: str):
    cfg = json.loads(base_cfg.read_text())

    # Try common schemas; adjust if your keys differ
    pri = cfg.setdefault("priors", {})
    f   = pri.setdefault("f", {})
    f["mean"]  = float(f.get("mean", 2.60))
    f["sigma"] = float(sigma_f)

    cfg["out"] = out_json
    cfg["out_prefix"] = out_json.rsplit(".json",1)[0]

    tmp = pathlib.Path("outputs")/f"tmp_cfg_sigmaf_{str(sigma_f).replace('.','p')}.json"
    tmp.write_text(json.dumps(cfg, indent=2))
    print(f"[run] σ_f={sigma_f} → {out_json}")
    sp.check_call(["python3","run_joint_bao_sn.py", str(tmp)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="Path to your working JSON config")
    args = ap.parse_args()
    base = pathlib.Path(args.base_config)
    if not base.exists():
        raise SystemExit(f"Missing: {base}")
    pathlib.Path("outputs").mkdir(exist_ok=True)
    bump_sigma_and_run(base, 0.10, "outputs/joint_gamma_sigmaf_0p10.json")
    bump_sigma_and_run(base, 0.20, "outputs/joint_gamma_sigmaf_0p20.json")
    print("[done] sensitivity runs finished.")
if __name__ == "__main__":
    main()
