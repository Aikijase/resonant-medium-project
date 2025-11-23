# Resonant Medium — Predictions (Lock Version)

This package declares two pre-registered, parameter-tied signatures derived from the Echo kernel.
Primary params sourced from Phase-20 fit if present (else Phase-21, else fallback).

## Kernel (effective)
A=1, tau_m=1, omega=1, phi=0
chi2(P20)=0.000 ; chi2(P8)=0.000

## P1: CMB/weak-lensing low-ω residual (template)
File: `lensing_residual_template.csv`
Definition: residual(l) = A·exp(-t(l)/tau_m)·cos(omega·t(l)+phi)
Mapping: t(l) = 0.003 · l (fixed)
Guardband window: ±10.0% around omega

## P2: BAO phase tweak (proxy; ablation)
File: `bao_phase_tweak_proxy.csv`
with_memory = base(z) + 1%·A·exp(-z/tau_m)·cos(omega·z+phi)
Ablation: tau_to_zero ≡ base(z)

Notes: These are registration-ready templates, not a cosmology pipeline.
