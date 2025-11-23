# shmf_counts.py
# Predict subhalo counts by scaling a ΛCDM baseline with a measured suppression ratio.
# Default: baseline_LCDM = 60 dwarfs in [1e9, 1e10] Msun/h for a MW-like host.

import json

def predict_counts_from_ratio(suppression_ratio,
                              baseline_LCDM=60.0,
                              Mhost=1e12,
                              Mmin=1e9,
                              Mmax=1e10):
    """
    Return a simple, calibrated prediction:
    - N_LCDM is your chosen baseline count for [Mmin, Mmax] around a MW-like host.
    - N_RES  = suppression_ratio * N_LCDM
    """
    if suppression_ratio is None:
        N_res = None
    else:
        N_res = suppression_ratio * baseline_LCDM

    return {
        "host_mass_Msun_h": Mhost,
        "mass_range_Msun_h": [Mmin, Mmax],
        "baseline_LCDM": baseline_LCDM,
        "pred_LCDM": baseline_LCDM,
        "pred_Resonant": N_res,
        "suppression_ratio_used": suppression_ratio
    }

if __name__ == "__main__":
    # tiny CLI for ad-hoc checks
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--baseline", type=float, default=60.0)
    ap.add_argument("--Mhost", type=float, default=1e12)
    ap.add_argument("--Mmin", type=float, default=1e9)
    ap.add_argument("--Mmax", type=float, default=1e10)
    args = ap.parse_args()
    out = predict_counts_from_ratio(args.ratio, args.baseline, args.Mhost, args.Mmin, args.Mmax)
    print(json.dumps(out, indent=2))
