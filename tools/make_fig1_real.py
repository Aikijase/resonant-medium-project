#!/usr/bin/env python3
"""
Generate a real-data Figure 1 for the Echo Equation paper.

This script:
  - loads Pantheon+ SN binned data (z, mu, sigma);
  - loads a corresponding SN model overlay (mu_model) as a baseline curve;
  - loads DESI DR1 BAO compressed DM points (kind,z,y_data,sigma);
  - produces a 2-panel figure:
        left  = SN Hubble diagram (data + model curve)
        right = BAO DM measurements (data only for now)
  - writes the figure to figures/fig1.png

You can later extend this to add a second (Echo) model curve by loading
a second overlay for SN and BAO and plotting it as a dashed line.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SN_CSV = ROOT / "data" / "pantheon_plus" / "sn_MATCHED_mu.csv"
SN_OVERLAY_CSV = ROOT / "outputs" / "sn_fit_stat_fullcov_dat_overlay.csv"

BAO_CSV = ROOT / "data" / "desi_dr1_bao" / "bao_measurements_long_interleaved_sigma.csv"

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = FIG_DIR / "fig1.png"


def load_sn_data():
    """
    Load Pantheon+ SN binned data from sn_MATCHED_mu.csv.

    CSV format (as confirmed by `head`):
        z,mu,sigma
    """
    sn = np.loadtxt(SN_CSV, delimiter=",", skiprows=1)
    z_sn = sn[:, 0]
    mu_data = sn[:, 1]
    mu_err = sn[:, 2]
    return z_sn, mu_data, mu_err


def load_sn_model_overlay():
    """
    Load SN model overlay from sn_fit_stat_fullcov_dat_overlay.csv.

    CSV header:
        z,mu,mu_model,log10_r

    We treat mu_model as a baseline/ΛCDM-like theory curve.
    """
    sn_ov = np.genfromtxt(SN_OVERLAY_CSV, delimiter=",", names=True)
    z_model = sn_ov["z"]
    mu_model = sn_ov["mu_model"]
    return z_model, mu_model


def load_bao_data():
    """
    Load DESI DR1 BAO compressed data from
    bao_measurements_long_interleaved_sigma.csv.

    CSV header (as confirmed by `head`):
        kind,z,y_data,sigma

    We restrict to kind == 'DM' for a distance-like measure.
    """
    bao = np.genfromtxt(BAO_CSV, delimiter=",", names=True, dtype=None, encoding=None)
    mask_dm = (bao["kind"] == "DM")
    z_bao = bao["z"][mask_dm]
    bao_data = bao["y_data"][mask_dm]
    bao_err = bao["sigma"][mask_dm]
    return z_bao, bao_data, bao_err


def make_figure():
    # ----- Load data -----
    z_sn, mu_data, mu_err = load_sn_data()
    z_sn_model, mu_model_sn = load_sn_model_overlay()
    z_bao, bao_data, bao_err = load_bao_data()

    # ----- Sanity checks -----
    # If the SN model redshift grid differs slightly, we just plot it as-is.
    # You can add interpolation here later if needed.
    print(f"[SN] Data points:   {len(z_sn)}")
    print(f"[SN] Model points:  {len(z_sn_model)}")
    print(f"[BAO] DM points:    {len(z_bao)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---------- Panel 1: SNe Hubble diagram ----------
    ax = axes[0]
    ax.errorbar(
        z_sn,
        mu_data,
        yerr=mu_err,
        fmt="o",
        ms=3,
        lw=1,
        capsize=2,
        label="Pantheon+ binned",
    )
    ax.plot(
        z_sn_model,
        mu_model_sn,
        lw=1.5,
        label=r"Baseline model (SN overlay)",
    )
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"Distance modulus $\mu$")
    ax.set_title("SNe Ia")
    ax.legend(fontsize=8)

    # ---------- Panel 2: DESI BAO DM measurements ----------
    ax = axes[1]
    ax.errorbar(
        z_bao,
        bao_data,
        yerr=bao_err,
        fmt="o",
        ms=4,
        lw=1,
        capsize=2,
        label="DESI BAO DR1 (DM)",
    )
    ax.set_xlabel(r"$z_{\rm eff}$")
    ax.set_ylabel(r"$y_{\rm DM}$ (compressed)")
    ax.set_title("BAO (DM only)")
    ax.legend(fontsize=8)

    fig.suptitle(r"BAO + SNe fits (data + baseline model)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"[OK] Wrote {OUT_FIG}")


if __name__ == "__main__":
    make_figure()
