#!/usr/bin/env python3
import argparse, json, re, pathlib, sys

def fm(x):
    m, s = x["mean"], x["sd"]
    if s >= 0.1: return f"{m:.2f} \\pm {s:.2f}"
    if s >= 0.01: return f"{m:.3f} \\pm {s:.3f}"
    return f"{m:.4f} \\pm {s:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--tex", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(pathlib.Path(args.fit).read_text(encoding="utf-8"))
    # Patterns expect rows like: "$\\tauMem$ & <NUM> & units"
    repls = [
        (r"\$\\tauMem\$\s*&\s*<NUM>",          f"$\\\\tauMem$ & {fm(d['tau'])}"),
        (r"\$\\kappaMem\$\s*&\s*<NUM>",        f"$\\\\kappaMem$ & {fm(d['kappa'])}"),
        (r"\$\\wzero\$\s*&\s*<NUM>",           f"$\\\\wzero$ & {fm(d['omega0'])}"),
        (r"\$\\Qfact\$\s*&\s*<NUM>",           f"$\\\\Qfact$ & {fm(d['Q'])}"),
        (r"\$C_\\mathrm\{cal\}\$\s*&\s*<NUM>", f"$C_\\\\mathrm{{cal}}$ & {fm(d['C_cal'])}"),
    ]

    txt = pathlib.Path(args.tex).read_text(encoding="utf-8")
    for pat, rep in repls:
        # Use lambda so re doesn't interpret backslashes in the replacement
        txt = re.sub(pat, (lambda m, rep=rep: rep), txt)

    pathlib.Path(args.out).write_text(txt, encoding="utf-8")
    print(f"filled: {args.out}")

if __name__ == "__main__":
    sys.exit(main())
