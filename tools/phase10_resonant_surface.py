#!/usr/bin/env python3
"""
Phase-10: Fit a tiny ridge surface to Phase-8 deltas and emit a forecast stub.
- Reads outputs/phase8/phase8_metrics.csv
- Builds a design for locks: tau0, kappa0, A0 (baseline is reference)
- Fits y = w_tau*tau0 + w_kap*kappa0 + w_A*A0  (ridge; no intercept needed)
- Writes:
    outputs/phase10/surface.json
    outputs/phase10/surface.txt
    outputs/phase10/delta_fit.png
"""
import os, csv, json, math
import numpy as np
import matplotlib.pyplot as plt

CSV_IN = "outputs/phase8/phase8_metrics.csv"
OUT_DIR = "outputs/phase10"

def load_deltas(path):
    rows = list(csv.DictReader(open(path)))
    base = float(rows[0]["metric"])
    out = []
    for r in rows:
        name = os.path.basename(r["stem"]).replace("joint_","")
        if not r["metric"]: continue
        y = float(r["metric"]) - base
        out.append((name, y))
    return out  # [('baseline',0.0), ('tau0',+5.5), ('kappa0',+2.2), ('A0',+15.8)]

def build_design(pairs):
    # One-hot locks: tau0, kappa0, A0 (baseline=all zeros)
    X, y, labels = [], [], []
    for name, dy in pairs:
        v = [0,0,0]  # [tau0, kappa0, A0]
        if name.lower().startswith("tau"):   v[0]=1
        if name.lower().startswith("kappa"): v[1]=1
        if name.upper().startswith("A0"):    v[2]=1
        X.append(v); y.append(dy); labels.append(name)
    return np.array(X, float), np.array(y, float), labels

def ridge_fit(X, y, alpha=1.0):
    # Closed-form ridge: w = (X^T X + αI)^-1 X^T y
    XT = X.T
    A = XT @ X + alpha*np.eye(X.shape[1])
    b = XT @ y
    w = np.linalg.solve(A, b)
    yhat = X @ w
    return w, yhat

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = load_deltas(CSV_IN)
    X, y, labels = build_design(pairs)
    w, yhat = ridge_fit(X, y, alpha=0.1)

    coefs = {"w_tau0": float(w[0]), "w_kappa0": float(w[1]), "w_A0": float(w[2])}
    J = {
        "design_labels": labels,
        "y_true": y.tolist(),
        "y_pred": yhat.tolist(),
        "weights": coefs,
        "note": "Ridge fit on lock indicators; baseline is 0. Positive weight = penalty when that loop is removed."
    }
    json.dump(J, open(f"{OUT_DIR}/surface.json","w"), indent=2)
    with open(f"{OUT_DIR}/surface.txt","w") as f:
        f.write("Phase-10 ridge weights (Δ penalty per lock):\n")
        for k,v in coefs.items(): f.write(f"  {k} = {v:.6g}\n")

    # simple plot
    names = ["tau0","kappa0","A0"]
    vals  = [coefs["w_tau0"], coefs["w_kappa0"], coefs["w_A0"]]
    plt.figure(figsize=(5,3.5))
    plt.bar(names, vals)
    plt.axhline(0, lw=1)
    plt.ylabel("Δ penalty (fitted)")
    plt.title("Phase-10: Penalty per lock (ridge)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/delta_fit.png", dpi=200)
    print("Wrote", f"{OUT_DIR}/surface.json")
    print("Wrote", f"{OUT_DIR}/surface.txt")
    print("Wrote", f"{OUT_DIR}/delta_fit.png")

if __name__ == "__main__":
    main()
