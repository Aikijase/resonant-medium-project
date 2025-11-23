#!/usr/bin/env python3
# tools/phase14_safe_zoom.py
import argparse, csv, os, sys, time, json, subprocess
from math import isfinite

def parse_args():
    p = argparse.ArgumentParser(description="Phase 14 safe zoom runner (resumable + timeout tolerant)")
    p.add_argument("--center-omega2", type=float, required=True)
    p.add_argument("--center-Kphi", type=float, required=True)
    p.add_argument("--omega2-span", type=float, default=0.12)
    p.add_argument("--Kphi-span", type=float, default=0.12)
    p.add_argument("--omega2-step", type=float, default=0.005)
    p.add_argument("--Kphi-step", type=float, default=0.005)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="p14_zoom_safe")
    p.add_argument("--steps", type=int, default=12000, help="inner sim --steps")
    p.add_argument("--burn_in", type=int, default=300, help="inner sim --burn_in")
    p.add_argument("--timeout", type=float, default=300.0, help="seconds per inner run")
    p.add_argument("--retries", type=int, default=1, help="retries per point on timeout/error")
    p.add_argument("--csv", type=str, default=None, help="results csv path (default: <outdir>/<prefix>_results.csv)")
    return p.parse_args()

def grid(v0, span, step):
    lo, hi = v0 - span/2.0, v0 + span/2.0
    n = int(round((hi - lo)/step)) + 1
    return [round(lo + i*step, 10) for i in range(n)]

def load_done(csv_path):
    done = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                k = (float(row["omega2"]), float(row["Kphi"]))
                done[k] = row
    return done

def write_header(csv_path):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["omega2","Kphi","status","sync_index","notes","stdout_tail","stderr_tail","elapsed_s","tries"])

def append_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)
        f.flush()

def tail(text, n=20):
    if text is None:
        return ""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="replace")
        except Exception:
            text = text.decode("latin-1", errors="replace")
    else:
        text = str(text)
    lines = text.splitlines()
    return "\n".join(lines[-n:])

def run_inner(args, w2, K, timeout_s):
    cmd = [
        sys.executable, "tools/phase10_phasecouple_demo.py",
        "--preset","neuron",
        "--omega2", str(w2),
        "--kv","0.10","--kx","0.20",
        "--Kphi", str(K),
        "--eps","0.06","--adapt_every","15","--noise","0.01",
        "--steps", str(args.steps),
        "--burn_in", str(args.burn_in),
        "--prefix", f"{args.prefix}_w2{w2:.4f}_K{K:.4f}"
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
        elapsed = time.time()-t0
        si = None
        notes = ""
        if p.returncode == 0:
            try:
                js = json.loads(p.stdout)
                si = js["metrics"]["sync_index"]
            except Exception as e:
                notes = f"json_parse_error: {e}"
        else:
            notes = f"returncode={p.returncode}"
        return ("ok" if si is not None and isfinite(si) else "error", si, notes, tail(p.stdout), tail(p.stderr), elapsed)
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        out = getattr(e, "output", None) or getattr(e, "stdout", b"")
        err = getattr(e, "stderr", b"")
        return ("timeout", None, "subprocess timeout", tail(out), tail(err), elapsed)

def main():
    args = parse_args()
    outcsv = args.csv or os.path.join(args.outdir, f"{args.prefix}_results.csv")
    write_header(outcsv)
    done = load_done(outcsv)

    W = grid(args.center_omega2, args.omega2_span, args.omega2_step)
    K = grid(args.center_Kphi,  args.Kphi_span,  args.Kphi_step)
    total = len(W)*len(K)
    idx = 0

    for w2 in W:
        for kphi in K:
            idx += 1
            key = (w2, kphi)
            if key in done and done[key].get("status") in ("ok","timeout","error"):
                continue
            tries = 0
            timeout_s = args.timeout
            status, si, notes, so_tail, se_tail, elapsed = ("", None, "", "", "", 0.0)
            while tries <= args.retries:
                status, si, notes, so_tail, se_tail, elapsed = run_inner(args, w2, kphi, timeout_s)
                if status == "timeout" and tries < args.retries:
                    tries += 1
                    timeout_s *= 1.5
                    continue
                break
            append_row(outcsv, [w2, kphi, status, ("" if si is None else f"{si:.6g}"), notes, so_tail, se_tail, f"{elapsed:.3f}", tries])
            if idx % 25 == 0:
                print(f"[safe-zoom] progress: {idx}/{total} (last={status}, w2={w2:.4f}, K={kphi:.4f})", flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
