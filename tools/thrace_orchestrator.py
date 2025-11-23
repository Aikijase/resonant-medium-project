from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from tools.thrace_recycling_model import RecyclingModel
    from tools.thrace_fit import fit_recycling_model
except ImportError:
    from thrace_recycling_model import RecyclingModel
    from thrace_fit import fit_recycling_model


def ensure_dirs():
    (_PROJECT_ROOT / "outputs/thrace").mkdir(parents=True, exist_ok=True)


def _normalize_rho_bh_columns(df, z_col=None, rho_bh_col=None):
    import pandas as pd
    def pick(name, aliases):
        for a in ([name] if name else []) + aliases:
            for c in df.columns:
                if c.lower() == a.lower():
                    return c
        return None

    z_found = pick("z", ["redshift"]) if not z_col else z_col
    rho_found = pick("rho_bh", [
        "rhobh", "rho_bh_msun_mpc3", "rho_bh_msun_per_mpc3",
        "rho_bh_msun/mpc^3", "rho_bh_msun/mpc3", "rho_bh_density"
    ]) if not rho_bh_col else rho_bh_col

    if z_found is None or rho_found is None:
        raise ValueError(f"Missing columns. Found: {list(df.columns)}")

    return df.rename(columns={z_found: "z", rho_found: "rho_bh"})[["z", "rho_bh"]]


def load_rho_bh(path: str, z_col=None, rho_bh_col=None):
    import pandas as pd
    df = pd.read_csv(path)
    return _normalize_rho_bh_columns(df, z_col=z_col, rho_bh_col=rho_bh_col)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--rho-bh", dest="rho_bh")
    p.add_argument("--z-col")
    p.add_argument("--rho-bh-col")
    p.add_argument("--out-prefix", default="outputs/thrace/recycling")
    p.add_argument("--alpha", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--beta-de", dest="beta_de", type=float)
    p.add_argument("--beta-dm", dest="beta_dm", type=float)
    args = p.parse_args()

    ensure_dirs()

    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config)) or {}

    path = args.rho_bh or cfg.get("rho_bh_path")
    if not path:
        sys.exit("ERROR: provide --rho-bh or set rho_bh_path in config")

    df = load_rho_bh(
        path,
        z_col=args.z_col or cfg.get("z_col"),
        rho_bh_col=args.rho_bh_col or cfg.get("rho_bh_col")
    )

    params = {
        "alpha": args.alpha or cfg.get("alpha", 1e-3),
        "gamma": args.gamma or cfg.get("gamma", 1e-3),
        "beta_de": args.beta_de or cfg.get("beta_de", 1e-2),
        "beta_dm": args.beta_dm or cfg.get("beta_dm", 1e-3),
    }

    model = RecyclingModel(**params)
    sol = model.solve_from_rho_bh(df)

    out = args.out_prefix
    sol.to_csv(f"{out}.solution.csv", index=False)
    fit = fit_recycling_model(df, sol, params)
    json.dump(fit, open(f"{out}.fit.json", "w"), indent=2)
    print("✅ Thrace run complete:", f"{out}.solution.csv", f"{out}.fit.json")


if __name__ == "__main__":
    main()
