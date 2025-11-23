#!/usr/bin/env python3
import json, os, numpy as np

def load_anchor(path="~/resonant-medium-project/configs/anchor_default.json"):
    """Return (A, w0, p, mapping) from the pinned config."""
    with open(os.path.expanduser(path)) as f:
        cfg = json.load(f)
    fp = cfg["fit_params"]
    return float(fp["A"]), float(fp["w0"]), float(fp["p"]), cfg.get("mapping","?")

def eps_curve(omega_hat, A, w0, p):
    """ε(ω̂) = A[1 - exp(-(ω̂/w0)^p)] with safe clamps."""
    x = np.clip(np.asarray(omega_hat, dtype=float), 1e-12, np.inf)
    w0 = max(float(w0), 1e-12)
    return A * (1.0 - np.exp(- (x / w0)**p))
