
# Thrace phase scripts (tools/)

## Installation / requirements
- Python >= 3.6
- **PEP 668 / Externally-managed env friendly**: use a virtual environment.

## Quick start (PEP 668-safe)
From project root:

```bash
# 1) Create a venv (first time only)
sudo apt-get update && sudo apt-get install -y python3-venv python3-dev  # if venv not available
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps inside the venv
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas pyyaml

# 3) (Optional) one-liner bootstrap
bash tools/thrace_bootstrap.sh
```

## Run
```bash
# Using YAML config
source .venv/bin/activate
PYTHONPATH=. python tools/thrace_orchestrator.py --config configs/thrace_default.yaml

# Or with explicit overrides
PYTHONPATH=. python tools/thrace_orchestrator.py       --rho-bh data/rho_bh_z.clean.csv       --alpha 0.001 --gamma 0.001 --beta-de 0.01 --beta-dm 0.001
```

## What to expect
- A solution CSV will be written to `outputs/thrace/` with columns `z, rho_de, rho_dm`.
- A fit JSON and simple report will be written too.

## Tests
Run the minimal tests:
```
source .venv/bin/activate
PYTHONPATH=. python -m tools.test_thrace
```

## Troubleshooting
- **Externally managed env (PEP 668)**: Always use a venv. Avoid `--break-system-packages` unless you know the trade-offs.
- **No module named tools.test_thrace**: Ensure these files are actually saved on disk in your repo and that `tools/__init__.py` exists. Run from project root with `PYTHONPATH=.` or inside the venv.
- **File not found**: Verify paths; `tools/thrace_orchestrator.py` must exist at that path.
- **Missing pyyaml** when using `--config`: `python -m pip install pyyaml` inside your venv.

## Extension notes
- Replace `z_to_time` with `astropy.cosmology` for accurate t(z) if desired.
- Swap `fit_recycling_model` with your true χ² pipeline when ready.
