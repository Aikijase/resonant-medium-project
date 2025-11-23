# Cosmic Recycling – Phase-21..23 Bundle
**Generated:** 2025-10-17 20:50

Drop-in scripts matching your protocol. Copy `tools/` into your repo and run as follows.

## Quickstart
```bash
# Phase-21: generate toy ρ_DE, ρ_DM, w_proxy from ρ_BH(z)
chmod +x tools/phase21_recycling_ode.py
python3 tools/phase21_recycling_ode.py --k 0.12 --alpha 1.1 --epsilon 0.7

# Phase-22: fit parameters to align w_proxy with Phase-8-like target
chmod +x tools/phase22_recycling_fit.py
python3 tools/phase22_recycling_fit.py

# Phase-23: produce predictions sheet (ΔNeff proxy, A_L proxy, fσ8 proxies)
chmod +x tools/phase23_predict_sheet.py
python3 tools/phase23_predict_sheet.py
```

## Files written
- `outputs/phase21/recycling_source.csv|txt|png`
- `outputs/phase22/recycling_fit.json|txt|png`
- `outputs/phase23/predictions.csv|txt`

## Notes
- Phase-21 will synthesize a smooth `rho_bh(z)` if you don’t pass `--bh-csv`. Point it at your compiled curve when ready.
- All models are toy/stubbed for wiring; replace proxies with your full equations as you iterate.
