
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def _prepare_ax(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def _save(fig, outpath):
    outpath = Path(outpath)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def _to_float_array(series_like):
    a = np.asarray(series_like)
    return a.astype(float)

def plot_fit_comparison(df_or_path, outpath):
    if not isinstance(df_or_path, pd.DataFrame):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    x = _to_float_array(df["x"])
    y_echo = _to_float_array(df["y_echo"])
    y_lcdm = _to_float_array(df["y_lcdm"])
    yerr_echo = _to_float_array(df["yerr_echo"]) if "yerr_echo" in df.columns else None
    yerr_lcdm = _to_float_array(df["yerr_lcdm"]) if "yerr_lcdm" in df.columns else None

    fig, ax = _prepare_ax(
        title="Model Fit Comparison (Echo vs ΛCDM)",
        xlabel="x (e.g., z or D_V index)",
        ylabel="Observable (arbitrary units)",
    )

    if yerr_echo is not None:
        ax.errorbar(x, y_echo, yerr=yerr_echo, fmt="o", label="Echo (±1σ)")
    else:
        ax.plot(x, y_echo, label="Echo", linewidth=2)

    if yerr_lcdm is not None:
        ax.errorbar(x, y_lcdm, yerr=yerr_lcdm, fmt="s", label="ΛCDM (±1σ)")
    else:
        ax.plot(x, y_lcdm, label="ΛCDM", linewidth=2, linestyle="--")

    # Δχ² annotation if present
    if hasattr(df, "attrs") and "delta_chi2" in df.attrs:
        ax.text(0.02, 0.95, f"Δχ² ≈ {df.attrs['delta_chi2']:.2f}", transform=ax.transAxes, va="top")

    ax.legend()
    _save(fig, outpath)

def plot_spectral_density(df_or_path, outpath):
    if not isinstance(df_or_path, pd.DataFrame):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    omega = _to_float_array(df["omega"])
    density = _to_float_array(df["density"])

    fig, ax = _prepare_ax(
        title="Operator Spectral Density",
        xlabel=r"$\omega$ (rad/e-fold)",
        ylabel="Density (arb.)",
    )

    ax.plot(omega, density, linewidth=2)

    # Highlight peak region if provided
    if "is_peak" in df.columns:
        mask = np.asarray(df["is_peak"]).astype(float) > 0.0
        if mask.any():
            ax.fill_between(omega[mask], density[mask], alpha=0.2)

    ax.grid(True, alpha=0.3)
    _save(fig, outpath)

def plot_lensing_residuals(df_or_path, outpath):
    if not isinstance(df_or_path, pd.DataFrame):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    ell = _to_float_array(df["ell"])
    residual = _to_float_array(df["residual"])
    residual_err = _to_float_array(df["residual_err"]) if "residual_err" in df.columns else None

    fig, ax = _prepare_ax(
        title="CMB Lensing Residuals",
        xlabel=r"Multipole $\ell$",
        ylabel=r"Fractional Residual $(\mathrm{model}/\mathrm{Planck}-1)$",
    )

    if residual_err is not None:
        ax.errorbar(ell, residual, yerr=residual_err, fmt="o", label="Residuals")
    else:
        ax.plot(ell, residual, label="Residuals", linewidth=2)

    ax.axhline(0.0, linestyle=":", linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, outpath)
